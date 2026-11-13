"""A wrapper that normalizes the observation by running statistics.

Paper-only helper of the MHD control experiments, not part of the FluidGym package.
"""

from collections.abc import Sequence

import torch
from gymnasium import spaces

from fluidgym.types import FluidEnvLike
from fluidgym.wrappers.fluid_wrapper import FluidWrapper


class NormalizeObservation(FluidWrapper):
    """Normalize a flat observation to zero mean and unit variance per entry.

    Mean and variance are running estimates over every observation seen in
    training mode and stay frozen in validation and test mode. To autograd they
    are constants: the update runs on the detached observation, and a new
    estimate replaces the old tensors instead of writing into them, so the graph
    of a BPTT window keeps the statistics its steps were normalized with. The
    normalization is affine in the observation, so the gradient passes through
    it scaled by ``1 / std``; only clipped entries lose their gradient.

    Parameters
    ----------
    env: FluidEnvLike
        The environment to wrap. Its observation space has to be a flat Box,
        e.g. that of :class:`FlattenObservation`.

    epsilon: float
        Added to the variance before the square root, and the initial count of
        the running estimate.

    clip: float | None
        Bound on the normalized observation, or None to leave it unbounded.
    """

    def __init__(
        self, env: FluidEnvLike, epsilon: float = 1e-8, clip: float | None = 10.0
    ) -> None:
        super().__init__(env)
        if not isinstance(self._env.observation_space, spaces.Box):
            raise ValueError(
                "NormalizeObservation only supports flat Box observation spaces; "
                "wrap the environment in FlattenObservation first."
            )

        shape = self._env.observation_space.shape
        self.__mean = torch.zeros(shape, dtype=torch.float64, device=env.cuda_device)
        self.__var = torch.ones(shape, dtype=torch.float64, device=env.cuda_device)
        self.__count = epsilon
        self.__epsilon = epsilon
        self.__clip = clip
        self.__update = True

    def __update_stats(self, obs: torch.Tensor) -> None:
        # The leading env (vectorized) and agent (MARL) dims are samples
        batch = obs.detach().to(torch.float64).reshape(-1, *self.__mean.shape)
        batch_count = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)

        # Parallel merge of the two estimates (Chan et al.); new tensors, never
        # in-place, since the graph of the current window holds the old ones
        delta = batch_mean - self.__mean
        total = self.__count + batch_count
        m2 = (
            self.__var * self.__count
            + batch_var * batch_count
            + delta.pow(2) * self.__count * batch_count / total
        )
        self.__mean = self.__mean + delta * batch_count / total
        self.__var = m2 / total
        self.__count = total

    def __normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if self.__update:
            self.__update_stats(obs)
        mean = self.__mean.to(obs.dtype)
        std = (self.__var + self.__epsilon).sqrt().to(obs.dtype)
        normalized = (obs - mean) / std
        if self.__clip is not None:
            normalized = normalized.clamp(-self.__clip, self.__clip)
        return normalized

    def train(self) -> None:
        """Set the environment to training mode, updating the statistics."""
        super().train()
        self.__update = True

    def val(self) -> None:
        """Set the environment to evaluation mode, freezing the statistics."""
        super().val()
        self.__update = False

    def test(self) -> None:
        """Set the environment to test mode, freezing the statistics."""
        super().test()
        self.__update = False

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        """The running statistics, to be saved next to the policy.

        Returns
        -------
        dict[str, torch.Tensor | float]
            The running mean, variance and sample count.
        """
        return {"mean": self.__mean, "var": self.__var, "count": self.__count}

    def load_state_dict(self, state: dict[str, torch.Tensor | float]) -> None:
        """Restore running statistics saved by :meth:`state_dict`.

        Parameters
        ----------
        state: dict[str, torch.Tensor | float]
            The running mean, variance and sample count.
        """
        self.__mean = torch.as_tensor(state["mean"]).to(self.__mean)
        self.__var = torch.as_tensor(state["var"]).to(self.__var)
        self.__count = float(state["count"])

    def reset(
        self,
        seed: int | None = None,
        randomize: bool | None = None,
        domain_idx: int | Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Resets the environment to an initial internal state, returning an initial
        observation and info.

        Parameters
        ----------
        seed: int | None
            The seed to use for random number generation. If None, the current seed is
            used.

        randomize: bool | None
            Whether to randomize the initial state. If None, the default behavior is
            used.

        domain_idx: int | Sequence[int] | None
            Index of the initial domain to load, one for all environments or one per
            environment of a vectorized environment. If None, the default behavior
            is used. Defaults to None.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            A tuple containing the normalized initial observation and an info
            dictionary.
        """
        obs, info = self._env.reset(
            seed=seed, randomize=randomize, domain_idx=domain_idx
        )
        return self.__normalize(obs), info

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        bool | torch.Tensor,
        bool | torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is
        necessary to call :meth:`reset` to reset this environment's state for the next
        episode.

        Parameters
        ----------
        action: torch.Tensor
            The action to take.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, bool, bool, dict[str, torch.Tensor]]
            A tuple containing the normalized observation, reward, terminated flag,
            truncated flag, and info dictionary.
        """
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self.__normalize(obs), reward, terminated, truncated, info
