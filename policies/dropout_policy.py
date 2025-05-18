import copy
import warnings
from typing import Optional, Union, List, Tuple
import torch as th
from torch import nn
from stable_baselines3.common.distributions import BernoulliDistribution, CategoricalDistribution, DiagGaussianDistribution, Distribution, MultiCategoricalDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BaseModel, ActorCriticPolicy
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs

from policies import utils


class DropoutActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_dropout_to_networks()
        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)

    def _apply_dropout_to_networks(self):
        """
        Apply dropout layers after each activation function in the respective networks,
        depending on whether the policy is CNN-based or MLP-based.
        """
        if isinstance(self.features_extractor, NatureCNN):
            # This is a CNN-based policy
            self.pi_features_extractor = copy.deepcopy(self.features_extractor)
            self.vf_features_extractor = copy.deepcopy(self.features_extractor)
            self.pi_features_extractor.cnn = self._add_dropout(self.pi_features_extractor.cnn, p=0.2)
            self.pi_features_extractor.linear = self._add_dropout(self.pi_features_extractor.linear, p=0.2)
            self.vf_features_extractor.cnn = self._add_dropout(self.vf_features_extractor.cnn, p=0.2)
            self.vf_features_extractor.linear = self._add_dropout(self.vf_features_extractor.linear, p=0.2)
        else:
            # This is an MLP-based policy
            self.mlp_extractor.policy_net = self._add_dropout(self.mlp_extractor.policy_net, p=0.3)
            self.mlp_extractor.value_net = self._add_dropout(self.mlp_extractor.value_net, p=0.3)

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
        return nn.Sequential(*modified_layers)

    def get_all_dropout_outputs(self, obs: th.Tensor) -> Tuple[Union[List[th.Tensor], List[Tuple[th.Tensor, th.Tensor]]], List[th.Tensor]]:
        """
        Computes multiple forward passes with dropout enabled to approximate an uncertainty estimate.

        :param obs: The observation input.
        :return: A tuple containing:
            - A list of action distributions from multiple stochastic forward passes.
            - A list of value outputs from multiple stochastic forward passes.
        """
        action_distributions, value_outputs = [], []
        num_samples = 7  # Number of stochastic passes
        self.train()  # Ensure dropout is active
        for _ in range(num_samples):
            # Forward pass with dropout
            _, values, _ = self.forward(obs, deterministic=False)
            action_distribution = self.get_distribution(obs)
            if isinstance(action_distribution, CategoricalDistribution):
                probabilities = action_distribution.distribution.probs
            elif isinstance(action_distribution, DiagGaussianDistribution) or isinstance(action_distribution, StateDependentNoiseDistribution):
                probabilities = (action_distribution.distribution.mean, action_distribution.distribution.stddev)
            elif isinstance(action_distribution, MultiCategoricalDistribution):
                probabilities = [dist.probs for dist in action_distribution.distribution]
            elif isinstance(action_distribution, BernoulliDistribution):
                probabilities = action_distribution.distribution.probs
            else:
                raise ValueError(f"Unsupported action distribution type: {type(action_distribution)}")
            action_distributions.append(probabilities)
            value_outputs.append(values)
        return action_distributions, value_outputs

    def compute_uncertainties(self, obs: PyTorchObs) -> Tuple[float, float, float]:
        """
        Computes the uncertainty measures for a given observation using all dropout outputs.

        :param obs: The input observation for which uncertainties should be computed.
        :return: Tuple (Policy-EU, Value-EU, Policy-AU)
        """
        action_outputs, value_outputs = self.get_all_dropout_outputs(obs)
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
    All methods following below are functionally identical to those of the superclass.
    They have been adjusted by replacing super().extract_features(...) with BasePolicy.extract_features(...)
    to ensure that the feature extraction logic from BasePolicy is used directly to preserve the original behavior
    """
    def extract_features(self, obs: PyTorchObs, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        if self.share_features_extractor:
            return BaseModel.extract_features(self, obs, self.features_extractor if features_extractor is None else features_extractor)
        else:
            if features_extractor is not None:
                warnings.warn("Provided features_extractor will be ignored because the features extractor is not shared.", UserWarning)
            pi_features = BaseModel.extract_features(self, obs, self.pi_features_extractor)
            vf_features = BaseModel.extract_features(self, obs, self.vf_features_extractor)
            return pi_features, vf_features

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        features = BaseModel.extract_features(self, obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)


    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        features = BaseModel.extract_features(self, obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
