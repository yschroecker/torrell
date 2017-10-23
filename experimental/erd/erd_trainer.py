import time
from typing import Sequence, Optional
import random

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
                 discriminator: torch.nn.Module, memory_policy: torch.nn.Module, minimum_population: int,
                 envs: Sequence[Environment[int]], trainer_config: TrainerConfig,
                 look_ahead: int = 1, batch_size: int = 1, memory_rate: float = 1):
        super().__init__(trainer_config)

        self._discriminator = discriminator
        self._memory_policy = memory_policy
        self._current_policy = trainer_config.policy_model
        self._buffers = trainers.ring_buffer.RingBufferCollection(
            memory_size, [trainer_config.state_dim, trainer_config.action_dim, 1, 1, trainer_config.state_dim,
                          trainer_config.action_dim],
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
        self._sample_rate = 4

    def _update_memory_pi(self, iteration: int, batch: Batch):
        memory_samples = self._buffers.sample(batch.states.shape[0])
        X = memory_samples[0]
        X = X.astype(np.float32)
        y = memory_samples[1]
        y_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), y)
        X_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), X)
        X_var = torch.autograd.Variable(X_tensor, requires_grad=False)
        y_var = torch.autograd.Variable(y_tensor, requires_grad=False)
        y_pred = self._memory_policy(X_var)
        loss = f.nll_loss(y_pred.squeeze(), y_var.squeeze())
        loss.backward()

        loss_tensor = loss.data
        if torch_util.module_is_cuda(self._discriminator):
            loss_tensor = loss_tensor.cpu()

        visualization.global_summary_writer.add_scalar('m loss', loss_tensor.numpy(), iteration)

    def _update_discriminator(self, iteration: int, batch: Batch):
        memory_samples = self._buffers.sample(batch.states.shape[0])
        # test_weights = self._get_importance_weights(memory_samples[0], memory_samples[1])
        X = np.concatenate([batch.states, memory_samples[0]])
        X = X.astype(np.float32)
        y = np.concatenate([np.zeros((batch.states.shape[0],), dtype=np.float32),
                            np.ones((batch.states.shape[0],), dtype=np.float32)])
        y_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), y)
        X_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), X)
        X_var = torch.autograd.Variable(X_tensor, requires_grad=False)
        y_var = torch.autograd.Variable(y_tensor, requires_grad=False)
        y_pred = self._discriminator(X_var)
        loss = f.binary_cross_entropy_with_logits(y_pred.squeeze(), y_var)
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

    def _do_updates(self, iteration: int, batch: Batch):
        super()._do_updates(iteration, batch)
        if self._buffers.size > self._batch_size and iteration % self._sample_rate == 0:
            self._update_discriminator(iteration, batch)
            # self._update_memory_pi(iteration, batch)

    def _train_new(self, iteration: int):
        samples = None
        while samples is None or samples.states.shape[0] < self._batch_size:
            trainer = random.choice(self._trainers)
            trainer_batch = trainer.get_batch()
            memorize = np.random.random() < self._memory_rate
            if memorize:
                self._buffers.extend(
                    trainer_batch.states, trainer_batch.actions, trainer_batch.intermediate_returns,
                    trainer_batch.bootstrap_weights, trainer_batch.bootstrap_states,
                    trainer_batch.bootstrap_actions
                )
            if samples is None:
                samples = trainer_batch
            else:
                samples = Batch(
                    np.concatenate((samples.states, trainer_batch.states)),
                    np.concatenate((samples.actions, trainer_batch.actions)),
                    np.concatenate((samples.intermediate_returns, trainer_batch.intermediate_returns)),
                    np.concatenate((samples.bootstrap_states, trainer_batch.bootstrap_states)),
                    np.concatenate((samples.bootstrap_actions, trainer_batch.bootstrap_actions)),
                    np.concatenate((samples.bootstrap_weights, trainer_batch.bootstrap_weights)),
                    None
                )
        self.reward_ema = self._trainers[0].reward_ema
        self.eval_reward_ema = self._trainers[0].eval_reward_ema
        if time.time() - self._last_print > 60:
            print(f"iteration {iteration}, eval_score {self._trainers[0].eval_score_ema}")
            self._last_print = time.time()
        return self.do_train(iteration, samples)

    def _train_memory(self, iteration):
        states, actions, rewards, bootstrap_weights, next_states, next_actions = \
            self._buffers.sample(self._batch_size)
        importance_weights = self._get_importance_weights(states, actions.squeeze())

        # noinspection PyUnresolvedReferences
        batch = Batch(
            states=states,
            actions=actions.squeeze(),
            intermediate_returns=rewards.squeeze(),
            bootstrap_weights=bootstrap_weights.squeeze(),
            bootstrap_states=next_states,
            bootstrap_actions=next_actions.squeeze(),
            importance_weights=importance_weights
        )

        visualization.global_summary_writer.add_scalar('IW min', np.min(importance_weights), iteration)
        visualization.global_summary_writer.add_scalar('IW max', np.max(importance_weights), iteration)
        return self.do_train(iteration, batch)

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            if iteration % self._sample_rate == 0 or self._buffers.size < self._minimum_population:
                desc = self._train_new(iteration)
            else:
                desc = self._train_memory(iteration)
            trange.set_description(desc)

    def _get_importance_weights(self, states: np.ndarray, actions: np.ndarray) -> Optional[np.ndarray]:
        states = states.astype(np.float32)
        states_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), states)
        states_var = torch.autograd.Variable(states_tensor, volatile=True)

        actions = actions.astype(np.int32)
        actions_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), actions)
        actions_var = torch.autograd.Variable(actions_tensor, volatile=True)

        # pi_ratio
        memory_probabilities = torch.exp(self._memory_policy(states_var).gather(
            dim=1, index=actions_var.unsqueeze(1)).squeeze()
        )
        policy_probabilities = torch.exp(self._current_policy.log_probability(states_var, actions_var.squeeze()))
        action_weights_var = policy_probabilities/memory_probabilities
        action_weights_tensor = action_weights_var.data
        if self._current_policy.is_cuda:
            action_weights_tensor = action_weights_tensor.cpu()
        action_weights = action_weights_tensor.numpy()

        # d_ratio
        y_var = self._discriminator(states_var)
        y_tensor = y_var.squeeze().data
        if self._current_policy.is_cuda:
            y_tensor = y_tensor.cpu()
        y = y_tensor.numpy()
        state_weights = 1 / np.exp(y) - 1
        # return np.ones((states.shape[0],), dtype=np.float32)
        return state_weights
        # return state_weights * action_weights


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
