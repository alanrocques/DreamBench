"""Probe registry."""

from typing import Dict, Type

from dreambench.probes.base import BaseProbe

PROBE_REGISTRY: Dict[str, Type[BaseProbe]] = {}


def register_probe(name: str, probe_cls: Type[BaseProbe]) -> None:
    PROBE_REGISTRY[name] = probe_cls


def get_probe(name: str) -> BaseProbe:
    """Instantiate and return a probe by name."""
    if name not in PROBE_REGISTRY:
        raise KeyError(
            f"Unknown probe '{name}'. Available: {list(PROBE_REGISTRY.keys())}"
        )
    return PROBE_REGISTRY[name]()


def _load_defaults() -> None:
    from dreambench.probes.reward_fidelity import RewardFidelityProbe
    from dreambench.probes.object_permanence import ObjectPermanenceProbe
    from dreambench.probes.physics_consistency import PhysicsConsistencyProbe
    from dreambench.probes.entity_integrity import EntityIntegrityProbe
    from dreambench.probes.temporal_coherence import TemporalCoherenceProbe

    register_probe("reward_fidelity", RewardFidelityProbe)
    register_probe("object_permanence", ObjectPermanenceProbe)
    register_probe("physics_consistency", PhysicsConsistencyProbe)
    register_probe("entity_integrity", EntityIntegrityProbe)
    register_probe("temporal_coherence", TemporalCoherenceProbe)


_load_defaults()
