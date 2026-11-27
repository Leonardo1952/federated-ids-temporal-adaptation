"""Federated training primitives."""

from distributed_drift_ids.federation.fedavg import (
    BinaryLogisticModel,
    ModelState,
    aggregate_weighted,
)

__all__ = ["BinaryLogisticModel", "ModelState", "aggregate_weighted"]
