"""Environment wrapper registry."""

from typing import Dict, Type

from dreambench.envs.base import BaseEnvWrapper

ENV_REGISTRY: Dict[str, Type[BaseEnvWrapper]] = {}


def register_env(name: str, wrapper_cls: Type[BaseEnvWrapper]) -> None:
    ENV_REGISTRY[name] = wrapper_cls


def get_env_wrapper(name: str) -> BaseEnvWrapper:
    """Instantiate and return an environment wrapper by name."""
    if name not in ENV_REGISTRY:
        raise KeyError(
            f"Unknown environment '{name}'. Available: {list(ENV_REGISTRY.keys())}"
        )
    return ENV_REGISTRY[name]()


def _load_defaults() -> None:
    from dreambench.envs.atari.wrapper import AtariEnvWrapper
    from dreambench.envs.minigrid.wrapper import MiniGridEnvWrapper
    from dreambench.envs.crafter.wrapper import CrafterEnvWrapper
    from dreambench.envs.mock.wrapper import MockEnvWrapper

    register_env("atari", AtariEnvWrapper)
    register_env("minigrid", MiniGridEnvWrapper)
    register_env("crafter", CrafterEnvWrapper)
    register_env("mock", MockEnvWrapper)


_load_defaults()
