import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import DDPG
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.td3.policies import TD3Policy, Actor
from typing import List, Tuple
import types


class DropoutDDPGPolicy(TD3Policy):

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.actor.mu = self._add_dropout(self.actor.mu, p=0.3)
        self.critic.qf0 = self._add_dropout(self.critic.qf0, p=0.3)
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @staticmethod
    def _add_dropout(module, p):
        """
        Add dropout layers after each activation function in the module.
        """
        modified_layers = []
        for layer in module:
            modified_layers.append(layer)
            if isinstance(layer, (nn.ReLU, nn.Tanh, nn.LeakyReLU, nn.ELU, nn.GELU, nn.Sigmoid, nn.Softmax)):
                modified_layers.append(nn.Dropout(p=p))
        if isinstance(modified_layers[-1], nn.Dropout):
            modified_layers.pop()
        return nn.Sequential(*modified_layers)

    def get_all_dropout_outputs(self, obs: th.Tensor) -> Tuple[List[th.Tensor], List[th.Tensor]]:
        """
        Perform multiple stochastic forward passes through the actor and critic using dropout to estimate uncertainty.

        :param obs: The observation input.
        :return: A tuple containing a list of sampled actions and a list of corresponding Q-value estimates from multiple stochastic forward passes.
        """
        self.actor.train()
        self.critic.train()
        actions, q_values = [], []
        num_samples = 7
        for _ in range(num_samples):
            with th.no_grad():
                action = self.actor(obs)
                actions.append(action)
                q_values.append(self.critic(obs, action))
        return actions, q_values

    def compute_uncertainties(self, obs: PyTorchObs) -> Tuple[float, float]:
        """
        Computes the epistemic uncertainty measures for a given observation using all dropout outputs.

        :param obs: The input observation for which uncertainties should be computed.
        :return: Tuple (Policy-EU, Value-EU)
        """
        action_outputs, q_values = self.get_all_dropout_outputs(obs)
        policy_eu = th.var(th.stack(action_outputs), dim=0).mean().item()
        value_eu = th.var(th.tensor([q[0].item() for q in q_values])).item()
        return policy_eu, value_eu


class EnsembleDDPGPolicy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_size = 7 # Number of ensemble heads
        self.active_head = 0  # Which head is currently active (round-robin)
        self._replace_with_ensemble_heads()

    def _replace_with_ensemble_heads(self) -> None:
        """
        Replace the standard actor.mu and critic.qf0 networks with ensemble versions.
        Each head is an independent copy of the original subnetwork.
        """
        self.actor.mu = nn.ModuleList([self.make_actor().mu for _ in range(self.ensemble_size)])
        self.actor_target.mu = nn.ModuleList([self.make_actor().mu for _ in range(self.ensemble_size)])
        self.critic.qf0 = nn.ModuleList([self.make_critic().qf0 for _ in range(self.ensemble_size)])
        self.critic_target.qf0 = nn.ModuleList([self.make_critic().qf0 for _ in range(self.ensemble_size)])
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=self._dummy_schedule(1), **self.optimizer_kwargs)
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=self._dummy_schedule(1), **self.optimizer_kwargs)

        policy_self = self

        def actor_forward(actor_self: Actor, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
            """
            Custom forward method for the actor network that routes the observation
            through the currently active ensemble head.

            :param actor_self: The dynamically bound Actor module (equivalent to self.actor).
            :param obs: The raw input observation tensor.
            :param deterministic: Unused in DDPG, included for compatibility.
            :return: The action tensor.
            """
            features = actor_self.extract_features(obs, actor_self.features_extractor)
            return actor_self.mu[policy_self.active_head](features)

        def critic_forward(critic_self: ContinuousCritic, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor]:
            """
            Custom forward method for the critic network that routes the observation-action pair
            through the currently active ensemble Q-head.

            :param critic_self: The dynamically bound ContinuousCritic module (equivalent to self.critic).
            :param obs: The raw input observations.
            :param actions: The action tensor.
            :return: A Tuple of the Q-value tensor.
            """
            with th.set_grad_enabled(not critic_self.share_features_extractor):
                features = critic_self.extract_features(obs, critic_self.features_extractor)
            return (critic_self.qf0[policy_self.active_head](th.cat([features, actions], dim=1)),)

        def q1_forward(critic_self: ContinuousCritic, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
            """
            Custom q1_forward method for the critic network that routes the observation-action pair
            through the currently active ensemble Q-head.

            :param critic_self: The dynamically bound ContinuousCritic module (equivalent to self.critic).
            :param obs: The raw input observations.
            :param actions: The input action tensor.
            :return: The Q-value tensor predicted by the active ensemble critic head.
            """
            with th.no_grad():
                features = critic_self.extract_features(obs, critic_self.features_extractor)
            return critic_self.qf0[policy_self.active_head](th.cat([features, actions], dim=1))

        self.actor.forward = types.MethodType(actor_forward, self.actor)
        self.actor_target.forward = types.MethodType(actor_forward, self.actor_target)
        self.critic.forward = types.MethodType(critic_forward, self.critic)
        self.critic_target.forward = types.MethodType(critic_forward, self.critic_target)
        self.critic.q1_forward = types.MethodType(q1_forward, self.critic)

    def get_all_ensemble_outputs(self, obs: th.Tensor) -> Tuple[List[th.Tensor], List[th.Tensor]]:
        """
        Perform forward passes through all ensemble actor and critic heads to estimate epistemic uncertainty.

        :param obs: The input observation tensor.
        :return: A tuple containing a list of actions predicted by each actor head and a list of  Q-values predicted by each critic head.
        """
        original_head = self.active_head
        self.actor.eval()
        self.critic.eval()
        actions, q_values = [], []
        with th.no_grad():
            for i in range(self.ensemble_size):
                self.active_head = i
                action = self.actor(obs)
                actions.append(action)
                q_values.append(self.critic(obs, action))
        self.active_head = original_head
        return actions, q_values

    def compute_uncertainties(self, obs: PyTorchObs) -> Tuple[float, float]:
        """
        Computes the epistemic uncertainty measures for a given observation using all policy and value ensemble heads.

        :param obs: The input observation for which uncertainties should be computed.
        :return: Tuple (Policy-EU, Value-EU)
        """
        action_outputs, q_values = self.get_all_ensemble_outputs(obs)
        policy_eu = th.var(th.stack(action_outputs), dim=0).mean().item()
        value_eu = th.var(th.tensor([q[0].item() for q in q_values])).item()
        return policy_eu, value_eu


class EnsembleDDPG(DDPG):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # === Target network prediction ===
            with th.no_grad():
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # === Critic update ===
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # === Actor update ===
            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                head = self.policy.active_head
                polyak_update(list(self.policy.actor.features_extractor.parameters()) + list(self.policy.actor.mu[head].parameters()),
                              list(self.policy.actor_target.features_extractor.parameters()) + list(self.policy.actor_target.mu[head].parameters()), self.tau)
                polyak_update(list(self.policy.critic.features_extractor.parameters()) + list(self.policy.critic.qf0[head].parameters()),
                              list(self.policy.critic_target.features_extractor.parameters()) + list(self.policy.critic_target.qf0[head].parameters()), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)
                self.policy.active_head = (self.policy.active_head + 1) % self.policy.ensemble_size

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
