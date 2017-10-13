from typing import Optional

import numpy as np
import tqdm
import torch.nn.functional as F
import torch

from environments.environment import Environment
from critic.temporal_difference import Batch
import trainers.ring_buffer
import trainers.online_trainer
import torch_util


class DiscreteExperienceReplay(trainers.online_trainer.DiscreteTrainer):
    def __init__(self, env: Environment[int], trainer_config: trainers.online_trainer.TrainerConfig, memory_size: int,
                 batch_size: int, initial_population: int, transition_batch_size: int=1):
        super().__init__(env, trainer_config)
        self._batch_size = batch_size
        self._buffers = trainers.ring_buffer.RingBufferCollection(
            memory_size, [trainer_config.state_dim, 1, 1, 1, trainer_config.state_dim, 1],
            dtypes=[np.float32, np.int32, np.float32, np.float32, np.float32, np.int32]
        )
        self._initial_population = initial_population
        self._transition_batch_size = transition_batch_size

    def train(self, num_iterations: int):
        trange = tqdm.trange(num_iterations)
        for iteration in trange:
            self._iterate(iteration, trange)

    def _iterate(self, iteration: int, trange: tqdm.tqdm):
        states, actions, rewards, terminal_states, next_states, next_actions = \
            self.collect_transitions(self._transition_batch_size)

        bootstrap_weights = self._discount_factor * (1 - np.array(terminal_states, dtype=np.float32))
        self._buffers.extend(states, actions, rewards, bootstrap_weights, next_states, next_actions)

        for _ in range(self._transition_batch_size):
            if self._buffers.size >= self._initial_population:
                states, actions, rewards, bootstrap_weights, next_states, next_actions = \
                    self._buffers.sample(self._batch_size)

                # noinspection PyUnresolvedReferences
                batch = Batch(
                    states=states,
                    actions=actions.squeeze(),
                    intermediate_returns=rewards.squeeze(),
                    bootstrap_weights=bootstrap_weights.squeeze(),
                    bootstrap_states=next_states,
                    bootstrap_actions=next_actions.squeeze(),
                    importance_weights=self._get_importance_weights(states, actions.squeeze())
                )
                trange.set_description(self.do_train(iteration, batch))

    def _get_importance_weights(self, states: np.ndarray, actions: np.ndarray) -> Optional[np.ndarray]:
        return None


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
                        indices[batch_start:batch_start+self._discriminator_batch_size]
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

                        loss = F.binary_cross_entropy_with_logits(y_pred.squeeze(), y_var)
                        self._discriminator_optimizer.zero_grad()
                        loss.backward()
                        self._discriminator_optimizer.step()
        #if total > 0:
            #print(1-incorrect/total)

        super()._iterate(*args)

    def _get_importance_weights(self, states: np.ndarray, actions: np.ndarray) -> Optional[np.ndarray]:
        X = self._samples_to_input(states, actions).astype(np.float32)
        X_tensor = torch_util.load_input(torch_util.module_is_cuda(self._discriminator), X)
        X_var = torch.autograd.Variable(X_tensor, volatile=True)
        y_var = self._discriminator(X_var)
        y = y_var.squeeze().data.numpy()
        weights = 1/np.exp(y) - 1
        return weights


