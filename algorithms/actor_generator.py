from typing import Sequence, Callable, Optional, NamedTuple

import math

import numpy as np
import scipy.stats
import torch
from torch.autograd import Variable

import policies.policy
import environments.environment
import trainers.data_collection
import trainers.synchronous
import algorithms.algorithm
import data
import torch_util
import trainers.history_augmented
import generative.wgan
import visualization
import critic.value_td
import generative
import generative.generative_predecessor_model


class GeneratorBatch(NamedTuple):
    context: np.ndarray
    sample: np.ndarray


class ActorGenerator(algorithms.algorithm.Algorithm):
    def __init__(self, *, envs: Sequence[environments.environment.Environment], state_dim: int, action_dim: int,
                 generative_model: generative.ImplicitGenerativeModel,
                 policy_model: policies.policy.PolicyModel,
                 policy_builder: Callable[[policies.policy.PolicyModel], policies.policy.Policy],
                 policy_learning_rate: float,
                 rl_batch_size: int, num_predecessors: int, discount_factor: float, policy_frequency: int=1,
                 mean_reward_update_rate: float,
                 entropy_regularization: float=0, reward_log_smoothing: float=1,
                 max_len: int=-1, gradient_clipping: Optional[float]=None,
                 use_experimental_baseline: bool=False, burnin: int=0,
                 baseline_network: Optional[torch.nn.Module]=None,
                 baseline_lr: float=0, bootstrap: bool=False, **kwargs):

        self._generative_model = generative_model
        self._policy_optimizer = torch.optim.RMSprop(policy_model.parameters, policy_learning_rate)
        self._mean_reward_lr = mean_reward_update_rate
        self._burnin = burnin

        self._cuda = policy_model.is_cuda
        self._policy_frequency = policy_frequency

        self._discount_factor = discount_factor
        self._rl_batch_size = rl_batch_size
        self._num_predecessors = num_predecessors

        config = trainers.data_collection.TrainerConfig(
            state_dim, discount_factor, reward_log_smoothing, action_dim, max_len
        )
        self.policy_model = policy_model
        self.policy = policy_builder(policy_model)
        self._trainers = [
            trainers.data_collection.Trainer(env, self.policy, config, collect_history=True)
            for env in envs
        ]
        self._synchronous = trainers.synchronous.ParallelAgents(
            self._trainers, 1, rl_batch_size,
            collect_batch=trainers.history_augmented.collect_batch_with_history
        )
        self._entropy_regularization = entropy_regularization
        self._mean_reward = 0
        self._surrogate_loss = 0
        self._use_experimental_baseline = use_experimental_baseline
        self._gradient_clipping = gradient_clipping
        self._baseline_network = baseline_network
        if baseline_network is not None:
            self._baseline_optimizer = torch.optim.RMSprop(baseline_network.parameters(), baseline_lr)

        self._predecessor_model = generative.generative_predecessor_model.GenerativePredecessorModel(
            generative_model, action_dim, state_dim, policy_model.action_type is np.int32,
            policy_model.is_cuda, num_predecessors, bootstrap
        )
        super().__init__(self._trainers, config, self.policy)

    def iterate(self, iteration: int):
        if iteration % 2000 == 0:
            print(self._mean_reward)
        batch = self._synchronous.collect_transitions()
        self._predecessor_model.iterate(iteration, batch)
        rl_batch = data.Batch([sequence.transition_sequence for sequence in batch.sequences], batch.discount_factor)
        rl_batch = rl_batch.to_tensor(self._cuda)

        if iteration == 0:
            self._mean_reward = rl_batch.rewards().mean()
        else:
            self._mean_reward += self._mean_reward_lr * (rl_batch.rewards().mean() - self._mean_reward)

        context_var = Variable(
            torch.cat([seq.states[1, None].repeat(self._num_predecessors, 1) for seq in rl_batch.sequences])
        )
        imagined_states, imagined_actions = self._predecessor_model.imagine_samples(context_var)
        log_probabilities = self.policy_model.log_probability(imagined_states, imagined_actions)

        rewards = rl_batch.rewards().expand((self._num_predecessors, self._rl_batch_size)).transpose(1, 0).contiguous()\
            .view(-1)
        rewards = Variable(rewards)
        if self._baseline_network is not None:
            baseline = self._baseline_network(imagined_states)
            baseline_loss = ((baseline.squeeze() - rewards)**2).mean()
            visualization.global_summary_writer.add_scalar("baseline loss", baseline_loss, iteration)
            self._baseline_optimizer.zero_grad()
            baseline_loss.backward(retain_graph=True)
            self._baseline_optimizer.step()
            baseline.detach_()
        else:
            baseline = self._mean_reward
        if iteration >= self._burnin:
            self._surrogate_loss -= (log_probabilities.squeeze() * (rewards - baseline.squeeze())).mean()
            if iteration % 1000 == 0:  # TODO: 1000
                visualization.global_summary_writer.add_histogram("rewards", rewards, iteration)
                visualization.global_summary_writer.add_histogram("baseline", baseline, iteration)
            if self._entropy_regularization > 0:
                self._surrogate_loss -= self._entropy_regularization * self.policy_model.entropy(context_var).mean()
            if iteration % self._policy_frequency == 0:
                self._surrogate_loss /= self._policy_frequency
                self._policy_optimizer.zero_grad()
                visualization.global_summary_writer.add_scalar("surrogate loss", self._surrogate_loss, iteration)
                self._surrogate_loss.backward()
                if self._gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm(self.policy_model.parameters, self._gradient_clipping, 'inf')

                self._policy_optimizer.step()
                self._surrogate_loss = 0
        if iteration % 100 == 0:
            self.policy_model.visualize(iteration)
