import time
from typing import Sequence, Optional

import numpy as np
import torch
import tqdm
from torch.nn import functional as f

import torch_util
import trainers.ring_buffer
import visualization
from critic.temporal_difference import Batch
from environments.environment import Environment
from trainers.experience_replay import DiscreteExperienceReplay
from trainers.online_trainer import DiscreteTrainerBase, TrainerConfig, DiscreteNstepTrainer


class ERDTrainer(DiscreteTrainerBase):
    def __init__(self, memory_size: int,
                 discriminator: torch.nn.Module, minimum_population: int,
                 envs: Sequence[Environment[int]], trainer_config: TrainerConfig,
                 look_ahead: int = 1, batch_size: int = 1, memory_rate: float = 1):
        super().__init__(trainer_config)

        self._discriminator = discriminator
        self._buffers = trainers.ring_buffer.RingBufferCollection(
            memory_size, [trainer_config.state_dim, 1, 1, 1, trainer_config.state_dim, 1],
            dtypes=[np.float32, np.int32, np.float32, np.float32, np.float32, np.int32]
        )
        self._minimum_population = minimum_population
        # noinspection PyProtectedMember
        self._trainers = [DiscreteNstepTrainer(
            env, trainer_config if i == 0
            else trainer_config._replace(evaluation_frequency=0), look_ahead) for i, env in enumerate(envs)]
        self._batch_size = batch_size
        self._memory_rate = memory_rate
        self._last_print = 0

    def _do_updates(self, iteration: int, batch: Batch):
        super()._do_updates(iteration, batch)
        if self._buffers.size >= self._minimum_population:
            memory_samples = self._buffers.sample(batch.states.shape[0])
            X = np.concatenate([batch.states, memory_samples[0]])
            X = X.astype(np.float32)
            y = np.concatenate([np.zeros((batch.states.shape[0],), dtype=np.float32),
                                np.ones((batch.states.shape[0],), dtype=np.float32)])
            y_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), y)
            X_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), X)
            X_var = torch.autograd.Variable(X_tensor, requires_grad=False)
            y_var = torch.autograd.Variable(y_tensor, requires_grad=False)
            y_pred = self._discriminator(X_var)
            loss = f.binary_cross_entropy_with_logits(y_pred.squeeze(), y_var) / 30
            loss.backward()

            y_pred_tensor = y_pred.squeeze().data
            loss_tensor = loss.data
            if torch_util.module_is_cuda(self._discriminator):
                y_pred_tensor = y_pred_tensor.cpu()
                loss_tensor = loss_tensor.cpu()
            incorrect = sum(np.abs(np.exp(y_pred_tensor.numpy()) - y))
            total = len(y)

            visualization.global_summary_writer.add_scalar('D acc', 1 - incorrect / total, iteration)
            visualization.global_summary_writer.add_scalar('D loss', loss_tensor.numpy(), iteration)

    def train(self, num_iterations: int):
        trange = tqdm.tqdm(total=num_iterations)
        iteration = 0
        while iteration < num_iterations:

            trainer_samples = [trainer.get_batch() for trainer in self._trainers]
            samples = trainer_samples[0]
            memorize = np.random.random() < self._memory_rate
            for trainer_batch in trainer_samples[1:]:
                if memorize:
                    self._buffers.extend(
                        trainer_batch.states, trainer_batch.actions, trainer_batch.intermediate_returns,
                        trainer_batch.bootstrap_weights, trainer_batch.bootstrap_states,
                        trainer_batch.bootstrap_actions
                    )
                samples = Batch(
                    np.concatenate((samples.states, trainer_batch.states)),
                    np.concatenate((samples.actions, trainer_batch.actions)),
                    np.concatenate((samples.intermediate_returns, trainer_batch.intermediate_returns)),
                    np.concatenate((samples.bootstrap_states, trainer_batch.bootstrap_states)),
                    np.concatenate((samples.bootstrap_actions, trainer_batch.bootstrap_actions)),
                    np.concatenate((samples.bootstrap_weights, trainer_batch.bootstrap_weights)),
                    None
                )

            indices = np.arange(samples.states.shape[0])
            np.random.shuffle(indices)

            for batch_start in range(0, samples.states.shape[0], self._batch_size):
                batch_indices = indices[batch_start:batch_start + self._batch_size]
                batch = Batch(
                    states=samples.states[batch_indices],
                    actions=samples.actions[batch_indices],
                    intermediate_returns=samples.intermediate_returns[batch_indices],
                    bootstrap_weights=samples.bootstrap_weights[batch_indices],
                    bootstrap_states=samples.bootstrap_states[batch_indices],
                    bootstrap_actions=samples.bootstrap_actions[batch_indices],
                )
                iteration += 1
                trange.update(1)
                self.reward_ema = self._trainers[0].reward_ema
                self.eval_reward_ema = self._trainers[0].eval_reward_ema
                trange.set_description(self.do_train(iteration, batch))
                if time.time() - self._last_print > 60:
                    print(f"iteration {iteration}, eval_score {self._trainers[0].eval_score_ema}")
                    self._last_print = time.time()


class DiscreteExperienceReplayWithDiscriminator(DiscreteExperienceReplay):
    def __init__(self, discriminator: torch.nn.Module, discriminator_optimizer: torch.optim.Optimizer,
                 discriminator_batch_size: int, discriminator_passes: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._discriminator = discriminator
        self._discriminator_optimizer = discriminator_optimizer
        self._discriminator_batch_size = discriminator_batch_size
        self._discriminator_passes = discriminator_passes

    def _samples_to_input(self, states, actions):
        actions_one_hot = np.zeros((actions.shape[0], self._num_actions))
        actions_one_hot[np.arange(actions.shape[0]), actions] = 1
        return np.hstack([states, actions_one_hot])

    def _iterate(self, *args):
        indices = np.arange(self._transition_batch_size)
        np.random.shuffle(indices)
        incorrect = 0
        total = 0
        if self._buffers.size >= self._initial_population:
            for _ in range(self._discriminator_passes):
                for batch_start in range(0, self._discriminator_batch_size, self._transition_batch_size):
                    batch_indices = self._transition_batch_size + \
                                    indices[batch_start:batch_start + self._discriminator_batch_size]
                    if len(batch_indices) > 0:
                        online_samples = self._buffers[batch_indices]
                        memory_samples = self._buffers.sample(batch_indices.shape[0])
                        X = np.concatenate([self._samples_to_input(online_samples[0], online_samples[1]),
                                            self._samples_to_input(memory_samples[0], memory_samples[1])])
                        X = X.astype(np.float32)
                        y = np.concatenate([np.zeros((online_samples[0].shape[0],), dtype=np.float32),
                                            np.ones((memory_samples[0].shape[0],), dtype=np.float32)])
                        y_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), y)
                        X_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), X)
                        X_var = torch.autograd.Variable(X_tensor, requires_grad=False)
                        y_var = torch.autograd.Variable(y_tensor, requires_grad=False)
                        y_pred = self._discriminator(X_var)

                        incorrect += sum(np.abs(np.exp(y_pred.squeeze().data.numpy()) - y))
                        total += len(y)

                        loss = f.binary_cross_entropy_with_logits(y_pred.squeeze(), y_var)
                        self._discriminator_optimizer.zero_grad()
                        loss.backward()
                        # noinspection PyArgumentList
                        self._discriminator_optimizer.step()
                        # if total > 0:
                        # print(1-incorrect/total)

        super()._iterate(*args)

    def _get_importance_weights(self, states: np.ndarray, actions: np.ndarray) -> Optional[np.ndarray]:
        X = self._samples_to_input(states, actions).astype(np.float32)
        X_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), X)
        X_var = torch.autograd.Variable(X_tensor, volatile=True)
        y_var = self._discriminator(X_var)
        y = y_var.squeeze().data.numpy()
        weights = 1 / np.exp(y) - 1
        return weights