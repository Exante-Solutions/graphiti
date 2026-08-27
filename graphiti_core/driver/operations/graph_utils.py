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

from collections import defaultdict

from pydantic import BaseModel


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


def label_propagation(
    projection: dict[str, list[Neighbor]], max_iterations: int = 100
) -> list[list[str]]:
    """Cluster nodes with deterministic asynchronous label propagation.

    Every node starts with its UUID as a stable label. UUIDs are sorted before
    every sweep so the result is independent of database and mapping iteration
    order. Updates are applied in place, allowing later nodes in a sweep to
    observe earlier changes and avoiding the period-two oscillation of
    synchronous LPA.

    The current label wins a maximum-weight tie. Otherwise the smallest stable
    label wins. A repeated state or the iteration cap raises instead of
    returning a clustering that has not converged.
    """
    if max_iterations < 1:
        raise ValueError('max_iterations must be at least 1')

    node_uuids = sorted(projection)
    community_map = {uuid: uuid for uuid in node_uuids}
    seen_states = {tuple(community_map[uuid] for uuid in node_uuids)}

    for _ in range(max_iterations):
        no_change = True

        for uuid in node_uuids:
            curr_community = community_map[uuid]

            community_candidates: dict[str, int] = defaultdict(int)
            for neighbor in projection[uuid]:
                community_candidates[community_map[neighbor.node_uuid]] += neighbor.edge_count

            if not community_candidates:
                continue

            max_weight = max(community_candidates.values())
            if (
                curr_community in community_candidates
                and community_candidates[curr_community] == max_weight
            ):
                new_community = curr_community
            else:
                new_community = min(
                    community
                    for community, weight in community_candidates.items()
                    if weight == max_weight
                )

            if new_community != curr_community:
                community_map[uuid] = new_community
                no_change = False

        if no_change:
            break

        state = tuple(community_map[uuid] for uuid in node_uuids)
        if state in seen_states:
            raise RuntimeError('label_propagation detected a repeated state before convergence')
        seen_states.add(state)
    else:
        raise RuntimeError(
            f'label_propagation failed to converge within {max_iterations} iterations'
        )

    community_cluster_map: dict[str, list[str]] = defaultdict(list)
    for uuid in node_uuids:
        community = community_map[uuid]
        community_cluster_map[community].append(uuid)

    return sorted(community_cluster_map.values(), key=lambda cluster: cluster[0])
