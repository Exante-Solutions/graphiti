"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections.abc import Callable

import pytest
from pydantic import BaseModel

from graphiti_core.driver.operations.graph_utils import (
    Neighbor as DriverNeighbor,
)
from graphiti_core.driver.operations.graph_utils import (
    label_propagation as driver_label_propagation,
)
from graphiti_core.utils.maintenance.community_operations import (
    Neighbor as MaintenanceNeighbor,
)
from graphiti_core.utils.maintenance.community_operations import (
    label_propagation as maintenance_label_propagation,
)

LabelPropagation = Callable[[dict[str, list], int], list[list[str]]]
NeighborType = type[BaseModel]

LABEL_PROPAGATION_IMPLS = [
    pytest.param(driver_label_propagation, DriverNeighbor, id='driver'),
    pytest.param(maintenance_label_propagation, MaintenanceNeighbor, id='maintenance'),
]


def _undirected(
    neighbor_cls: NeighborType,
    edges: list[tuple[str, str, int]],
    isolated: list[str] | None = None,
) -> dict[str, list]:
    projection: dict[str, list] = {}
    for left, right, weight in edges:
        projection.setdefault(left, []).append(neighbor_cls(node_uuid=right, edge_count=weight))
        projection.setdefault(right, []).append(neighbor_cls(node_uuid=left, edge_count=weight))
    for node_uuid in isolated or []:
        projection.setdefault(node_uuid, [])
    return projection


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_two_node_period_two_regression_converges(
    impl: LabelPropagation, neighbor_cls: NeighborType
) -> None:
    projection = _undirected(neighbor_cls, [('node-a', 'node-b', 1)])

    assert impl(projection) == [['node-a', 'node-b']]


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_topology_is_invariant_to_projection_and_neighbor_permutations(
    impl: LabelPropagation, neighbor_cls: NeighborType
) -> None:
    edges = [
        ('hub', 'left', 2),
        ('hub', 'right', 2),
        ('left', 'leaf-a', 1),
        ('right', 'leaf-b', 1),
    ]
    forward = _undirected(neighbor_cls, edges)
    reverse_source = _undirected(neighbor_cls, list(reversed(edges)))
    permuted = {
        node_uuid: list(reversed(reverse_source[node_uuid]))
        for node_uuid in reversed(tuple(reverse_source))
    }

    assert impl(forward) == impl(permuted) == [['hub', 'leaf-a', 'leaf-b', 'left', 'right']]


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_isolated_nodes_stay_sorted_singletons(
    impl: LabelPropagation, neighbor_cls: NeighborType
) -> None:
    projection = _undirected(neighbor_cls, [], isolated=['node-c', 'node-a', 'node-b'])

    assert impl(projection) == [['node-a'], ['node-b'], ['node-c']]


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_disconnected_components_stay_separate(
    impl: LabelPropagation, neighbor_cls: NeighborType
) -> None:
    projection = _undirected(
        neighbor_cls,
        [
            ('a', 'b', 1),
            ('b', 'c', 1),
            ('a', 'c', 1),
            ('x', 'y', 1),
            ('y', 'z', 1),
            ('x', 'z', 1),
        ],
    )

    assert impl(projection) == [['a', 'b', 'c'], ['x', 'y', 'z']]


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_complete_graph_collapses_to_one_cluster(
    impl: LabelPropagation, neighbor_cls: NeighborType
) -> None:
    nodes = ['d', 'b', 'a', 'c']
    edges = [(left, right, 1) for index, left in enumerate(nodes) for right in nodes[index + 1 :]]

    assert impl(_undirected(neighbor_cls, edges)) == [['a', 'b', 'c', 'd']]


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_tie_policy_self_sticks_then_uses_smallest_stable_label(
    impl: LabelPropagation, neighbor_cls: NeighborType
) -> None:
    projection = _undirected(
        neighbor_cls,
        [
            ('a', 'c', 1),
            ('a', 'd', 1),
            ('b', 'd', 1),
        ],
    )

    assert impl(projection) == [['a', 'c'], ['b', 'd']]


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_iteration_cap_raises_instead_of_returning_partial_clusters(
    impl: LabelPropagation, neighbor_cls: NeighborType
) -> None:
    projection = _undirected(neighbor_cls, [('node-a', 'node-b', 1)])

    with pytest.raises(RuntimeError, match='failed to converge within 1 iterations'):
        impl(projection, max_iterations=1)


@pytest.mark.parametrize('impl, neighbor_cls', LABEL_PROPAGATION_IMPLS)
def test_repeated_state_cycle_raises_instead_of_returning_partial_clusters(
    impl: LabelPropagation, neighbor_cls: NeighborType
) -> None:
    projection = {
        'node-a': [neighbor_cls(node_uuid='node-b', edge_count=1)],
        'node-b': [neighbor_cls(node_uuid='node-c', edge_count=1)],
        'node-c': [neighbor_cls(node_uuid='node-a', edge_count=1)],
    }

    with pytest.raises(RuntimeError, match='detected a repeated state'):
        impl(projection)
