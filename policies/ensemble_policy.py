import numpy as np
import torch as th
from torch import nn

import warnings
from functools import partial
from typing import Optional, Union, Tuple

from stable_baselines3.common.policies import BaseModel, ActorCriticPolicy
from stable_baselines3.common.distributions import BernoulliDistribution, CategoricalDistribution, DiagGaussianDistribution, Distribution, MultiCategoricalDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from policies import utils


class EnsembleActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, ensemble_size=3, **kwargs):
        """
        Extension of ActorCriticPolicy to create multiple policy and value networks.
        Instead of a single actor-critic network, multiple networks are used,
        and the active head is cycled after each optimization step.

        :param ensemble_size: Number of actor-critic networks in the ensemble.
        """
        self.ensemble_size = ensemble_size  # Number of networks in the ensemble
        self.active_head = 0  #  By default, the first network is active
        super().__init__(*args, **kwargs)  # Call the original constructor
        self.features_dim = super().make_features_extractor().features_dim # explicitly for method _build_mlp_extractor
        self.optimizer_class = th.optim.Adam # explicitly for method _build
        self.original_optimizer_step = self.optimizer.step # Stores the actual PyTorch optimizer step.
        self.optimizer.step = self.optimizer_step # Ensures that our custom optimizer_step() is called instead of the default Adam optimizer.step, allowing us to switch the active head after each parameter update.

    def optimizer_step(self):
        """
        Performs an optimization step and then cyclically switches to the next head in the ensemble.
        """
        self.original_optimizer_step()  # Standard optimization step
        self.active_head = (self.active_head + 1) % self.ensemble_size  # Switching the active head after each backpropagation step

    def _build_mlp_extractor(self):
        """
        Creates multiple MLP extractors for the policy and value function.
        Each extractor uses the same network architecture.
        """
        self.mlp_extractors = nn.ModuleList([MlpExtractor(self.features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device) for _ in range(self.ensemble_size)])

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the ensemble networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()  # Creates all MLP extractors
        self.action_nets = nn.ModuleList()
        self.value_nets = nn.ModuleList()
        self.log_stds = nn.ParameterList()

        for i in range(self.ensemble_size):
            latent_dim_pi = self.mlp_extractors[i].latent_dim_pi  # Uses the respective MLP extractor
            if isinstance(self.action_dist, DiagGaussianDistribution):
                action_net, log_std = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi, log_std_init=self.log_std_init)
                self.log_stds.append(nn.Parameter(log_std))
            elif isinstance(self.action_dist, StateDependentNoiseDistribution):
                action_net, log_std = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init)
                self.log_stds.append(nn.Parameter(log_std))
            elif isinstance(self.action_dist, CategoricalDistribution):
                action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
            else:
                raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
            value_net = nn.Linear(self.mlp_extractors[i].latent_dim_vf, 1)
            self.action_nets.append(action_net)
            self.value_nets.append(value_net)

        if self.ortho_init:
            for i in range(self.ensemble_size):
                module_gains = {self.features_extractor: np.sqrt(2), self.mlp_extractors[i]: np.sqrt(2), self.action_nets[i]: 0.01, self.value_nets[i]: 1}
                if not self.share_features_extractor:
                    del module_gains[self.features_extractor]
                    module_gains[self.pi_features_extractor] = np.sqrt(2)
                    module_gains[self.vf_features_extractor] = np.sqrt(2)
                for module, gain in module_gains.items():
                    module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer, that contains all parameters of the ensemble networks, with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def compute_uncertainties(self, obs: PyTorchObs) -> Tuple[float, float, float]:
        """
        Computes the uncertainty measures for a given observation using all ensemble heads.

        :param obs: The input observation for which uncertainties should be computed.
        :return: Tuple (Policy-EU, Value-EU, Policy-AU)
        """
        action_outputs, value_outputs = utils.get_all_ensemble_head_outputs(self, obs, deterministic=False)
        if isinstance(self.action_dist, DiagGaussianDistribution) or isinstance(self.action_dist, StateDependentNoiseDistribution):
            policy_eu = utils.calculate_policy_eu_continuous(action_outputs)
            policy_au = utils.calculate_policy_au_continuous(action_outputs)
        elif isinstance(self.action_dist, CategoricalDistribution):
            policy_eu = utils.calculate_policy_epistemic_uncertainty(action_outputs)
            policy_au = utils.calculate_policy_total_uncertainty(action_outputs)
        else:
            raise NotImplementedError(f"Uncertainty estimation for distribution {type(self.action_dist)} not implemented")
        value_eu = utils.calculate_value_epistemic_uncertainty(value_outputs)
        return policy_eu, value_eu, policy_au

    """
    All following methods below are functionally identical to those of the superclass.
    They have been adjusted to work with the modified instance parameters of this class,
    e.g. self.mlp_extractors[self.active_head] instead of self.mlp_extractor, etc.
    """
    def reset_noise(self, n_envs: int = 1) -> None:
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_stds[self.active_head], batch_size=n_envs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractors[self.active_head](features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractors[self.active_head].forward_actor(pi_features)
            latent_vf = self.mlp_extractors[self.active_head].forward_critic(vf_features)
        values = self.value_nets[self.active_head](latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def extract_features(self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        if self.share_features_extractor:
            return BaseModel.extract_features(self, obs, self.features_extractor if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn("Provided features_extractor will be ignored because the features extractor is not shared.", UserWarning)
            pi_features = BaseModel.extract_features(self, obs, self.pi_features_extractor)
            vf_features = BaseModel.extract_features(self, obs, self.vf_features_extractor)
            return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        mean_actions = self.action_nets[self.active_head](latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_stds[self.active_head])
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_stds[self.active_head], latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractors[self.active_head](features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractors[self.active_head].forward_actor(pi_features)
            latent_vf = self.mlp_extractors[self.active_head].forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_nets[self.active_head](latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        features = BaseModel.extract_features(self, obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractors[self.active_head].forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        features = BaseModel.extract_features(self, obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractors[self.active_head].forward_critic(features)
        return self.value_nets[self.active_head](latent_vf)
