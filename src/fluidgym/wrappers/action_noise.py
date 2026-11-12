"""A wrapper that adds Gaussian noise to the actions."""

import torch

from fluidgym.types import FluidEnvLike
from fluidgym.wrappers.fluid_wrapper import FluidWrapper


class ActionNoise(FluidWrapper):
    """A wrapper that adds Gaussian noise to the actions.

    It adds Gaussian noise with standard deviation `sigma` to the actions and clips
    the noisy actions to the action space bounds, since the environment rejects
    actions outside of them.

    Parameters
    ----------
    env: FluidEnvLike
        The environment to wrap.

    sigma: float
        The standard deviation of the Gaussian noise.

    seed: int
        The random seed for the noise generator.
    """

    def __init__(self, env: FluidEnvLike, sigma: float, seed: int) -> None:
        super().__init__(env)
        self.__sigma = sigma
        self.__torch_rng = torch.Generator(device=self._env.cuda_device).manual_seed(
            seed
        )

    def step(
        self, action: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
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
        tuple[
        dict[str, torch.Tensor], torch.Tensor, bool, bool, dict[str, torch.Tensor]]
            A tuple containing the observation, reward, terminated flag, truncated flag,
            and info dictionary.
        """
        noisy_action = (
            action
            + torch.randn(
                size=action.shape,
                generator=self.__torch_rng,
                device=self._env.cuda_device,
                dtype=action.dtype,
            )
            * self.__sigma
        )

        # Per-agent bounds, broadcast over leading env and agent dims
        low = torch.as_tensor(
            self._env.action_space.low, device=action.device, dtype=action.dtype
        )
        high = torch.as_tensor(
            self._env.action_space.high, device=action.device, dtype=action.dtype
        )
        return self._env.step(torch.minimum(torch.maximum(noisy_action, low), high))
