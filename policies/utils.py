import torch as th
from typing import List, Tuple
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.distributions import (BernoulliDistribution, CategoricalDistribution,
                                                    DiagGaussianDistribution, MultiCategoricalDistribution,
                                                    StateDependentNoiseDistribution)


def get_all_ensemble_head_outputs(policy: ActorCriticPolicy, obs: PyTorchObs, deterministic: bool = False) -> Tuple[List[th.Tensor], List[th.Tensor]] | None:
    """
    Computes the probability distributions of all action and value heads for the given observation,
    independently of the currently active head.

    :param policy: The policy object (should be an instance of EnsembleActorCriticPolicy).
    :param obs: The observation input for which all heads should be evaluated.
    :param deterministic: Whether to use deterministic actions or not.
    :return: A tuple containing:
        - A list of probability distributions from all policy heads (normalized to sum to 1 where applicable).
        - A list of value outputs from all value heads.
    """
    action_distributions = []
    value_outputs = []

    # Store the currently active head
    current_active_head = policy.active_head

    try:
        for i in range(policy.ensemble_size):
            # Temporarily set the active head to i
            policy.active_head = i

            # Use the existing forward method to get values
            _, values, _ = policy.forward(obs, deterministic=deterministic)

            # Retrieve the action distribution
            action_distribution = policy.get_distribution(obs)

            # Use `probs` directly when available
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
    finally:
        # Restore the original active head
        policy.active_head = current_active_head

    return action_distributions, value_outputs


def calculate_policy_epistemic_uncertainty(action_probs: List[th.Tensor]) -> float:
    """
    Calculates the epistemic uncertainty of the policy by computing the variance
    of action probabilities across ensemble heads.
    The uncertainty is defined as the variance of the action heads with the highest mean probability.

    :param action_probs: List of tensors containing action probability distributions from all heads.
    :return: Epistemic uncertainty score (scalar).
    """
    action_probs = th.stack(action_probs)
    mean_probs = th.mean(action_probs, dim=0)
    max_action_index = th.argmax(mean_probs).item()
    selected_action_probs = action_probs[:, 0, max_action_index]
    return th.var(selected_action_probs).item()


def calculate_value_epistemic_uncertainty(value_outputs: List[th.Tensor]) -> float:
    """
    Computes the epistemic uncertainty for the value outputs by calculating the variance across ensemble heads.

    :param value_outputs: List of tensors containing value estimates from all heads.
    :return: Epistemic uncertainty score for the value function.
    """
    return th.var(th.stack(value_outputs)).item()


def calculate_policy_total_uncertainty(action_probs: List[th.Tensor]) -> float:
    """
    Computes the total uncertainty of the policy by calculating the average entropy of all policy heads.

    :param action_probs: List of tensors containing action probability distributions from all heads.
    :return: Total policy uncertainty score (scalar).
    """
    action_probs = th.stack(action_probs)
    entropies = -th.sum(action_probs * th.log(action_probs + 1e-8), dim=-1)
    return th.mean(entropies).item()


def calculate_policy_eu_continuous(action_outputs: List[Tuple[th.Tensor, th.Tensor]]) -> float:
    """
    Computes the epistemic uncertainty of the policy for continuous action spaces (Box),
    by calculating the variance of the means across all ensemble heads.

    :param action_outputs: List of (mean, std) tuples from all ensemble heads
    :return: Epistemic uncertainty of the policy
    """
    means = th.stack([mean.view(-1) for mean, _ in action_outputs])
    return th.mean(th.var(means, dim=0)).item()


def calculate_policy_au_continuous(action_outputs: List[Tuple[th.Tensor, th.Tensor]]) -> float:
    """
    Computes the aleatoric uncertainty of the policy for continuous action spaces (Box),
    by calculating the mean of the standard deviations across all actions and ensemble heads.

    :param action_outputs: List of (mean, std) tuples from all ensemble heads
    :return: Aleatoric uncertainty of the policy
    """
    stds = th.stack([std.view(-1) for _, std in action_outputs])
    return th.mean(stds).item()
