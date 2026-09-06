"""Exact, canonical solver for AYTOrakel.

Each solution is a bipartite graph.  Rows are participants in ``matrix.rows``;
every integer cell is a bit mask containing all matches from ``matrix.bits``.
There is exactly one array row per distinct Perfect Match graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import itertools
import json
import math
from typing import Any, Iterable, Iterator

import numpy as np


SCHEMA_VERSION = 2
SOLVER_VERSION = 4
MAX_COMBINATIONS_IN_POST = 24


class DataValidationError(ValueError):
    """Raised when season data cannot describe a valid calculation."""


class NoSolutionsError(RuntimeError):
    """Raised when an event contradicts all remaining solutions."""


@dataclass
class Snapshot:
    week: int
    event_number: int
    event_index: int
    event_type: str
    title: str
    solution_count: int
    probabilities: np.ndarray
    row_names: tuple[str, ...]
    bit_names: tuple[str, ...]
    active: dict[str, tuple[str, ...]]
    face_layer: str
    before_count: int = 0
    light_distribution: np.ndarray | None = None
    actual_lights: int | None = None
    combination_sample: np.ndarray | None = None


@dataclass
class NightRecord:
    week: int
    event_number: int
    distribution: np.ndarray
    actual_lights: int


@dataclass
class EventCountRecord:
    week: int
    event_number: int
    event_index: int
    event_type: str
    before_count: int
    after_count: int


@dataclass
class SolverState:
    season: str
    row_group: str
    bit_group: str
    groups: dict[str, list[str]]
    active: dict[str, set[str]]
    multi_matches: list[dict[str, Any]]
    face_layer: str
    solutions: np.ndarray | None = None
    processed_events: int = 0
    completed_week: int = 0
    night_records: list[NightRecord] = field(default_factory=list)
    event_records: list[EventCountRecord] = field(default_factory=list)

    @property
    def row_names(self) -> list[str]:
        return self.groups[self.row_group]

    @property
    def bit_names(self) -> list[str]:
        return self.groups[self.bit_group]

    @property
    def mask_dtype(self) -> np.dtype:
        return mask_dtype(len(self.bit_names))


class _SnapshotAccumulator:
    def __init__(self, n_rows: int, n_bits: int) -> None:
        self.count = 0
        self.edge_counts = np.zeros((n_rows, n_bits), dtype=np.uint64)
        self.sample: list[np.ndarray] = []

    def add(self, solutions: np.ndarray) -> None:
        if len(solutions) == 0:
            return
        self.count += len(solutions)
        for bit_index in range(self.edge_counts.shape[1]):
            bit = np.array(1 << bit_index, dtype=solutions.dtype)
            counts = np.count_nonzero(solutions & bit, axis=0).astype(np.uint64)
            self.edge_counts[:, bit_index] += counts
        remaining = MAX_COMBINATIONS_IN_POST + 1 - sum(len(x) for x in self.sample)
        if remaining > 0:
            self.sample.append(solutions[:remaining].copy())

    def probabilities(self) -> np.ndarray:
        if self.count == 0:
            raise NoSolutionsError("No possible Perfect Match graph remains")
        return self.edge_counts.astype(np.float64) * (100.0 / self.count)

    def combination_sample(self) -> np.ndarray | None:
        if self.count > MAX_COMBINATIONS_IN_POST or not self.sample:
            return None
        return np.concatenate(self.sample, axis=0)[: self.count]


def mask_dtype(bit_count: int) -> np.dtype:
    if bit_count <= 16:
        return np.dtype(np.uint16)
    if bit_count <= 32:
        return np.dtype(np.uint32)
    if bit_count <= 64:
        return np.dtype(np.uint64)
    raise DataValidationError("At most 64 participants are supported in the bit group")


@lru_cache(maxsize=None)
def faster_permutations(n: int) -> np.ndarray:
    """Return all permutations using a compact Fortran-ordered uint8 array."""
    if n < 1:
        return np.empty((1, 0), dtype=np.uint8)
    if n > 10:
        raise DataValidationError(
            "Exact full permutation generation currently supports at most 10 match slots"
        )
    perms = np.empty((math.factorial(n), n), dtype=np.uint8, order="F")
    perms[0, 0] = 0
    rows_to_copy = 1
    for i in range(1, n):
        perms[:rows_to_copy, i] = i
        for j in range(1, i + 1):
            start_row = rows_to_copy * j
            end_row = rows_to_copy * (j + 1)
            splitter = i - j
            perms[start_row:end_row, splitter] = i
            perms[start_row:end_row, :splitter] = perms[
                :rows_to_copy, :splitter
            ]
            perms[start_row:end_row, splitter + 1 : i + 1] = perms[
                :rows_to_copy, splitter:i
            ]
        rows_to_copy *= i + 1
    return perms


def _multi_match_partitions(
    bit_names: list[str], rules: list[dict[str, Any]], expected_groups: int
) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Yield unique partitions of bit participants into multi groups and singles."""
    name_to_index = {name: index for index, name in enumerate(bit_names)}
    normalized: list[tuple[int, tuple[int, ...]]] = []
    for rule in rules:
        size = int(rule["size"])
        known = tuple(sorted(name_to_index[name] for name in rule.get("known_members", [])))
        normalized.append((size, known))

    seen: set[tuple[tuple[int, ...], ...]] = set()

    def choose(rule_index: int, used: frozenset[int], groups: list[tuple[int, ...]]) -> None:
        if rule_index == len(normalized):
            singles = [(i,) for i in range(len(bit_names)) if i not in used]
            complete = tuple(sorted(groups) + singles)
            if len(complete) == expected_groups and complete not in seen:
                seen.add(complete)
                yielded.append(complete)
            return
        size, known = normalized[rule_index]
        if any(i in used for i in known):
            return
        available = [i for i in range(len(bit_names)) if i not in used and i not in known]
        for rest in itertools.combinations(available, size - len(known)):
            group = tuple(sorted(known + rest))
            choose(rule_index + 1, used.union(group), groups + [group])

    yielded: list[tuple[tuple[int, ...], ...]] = []
    choose(0, frozenset(), [])
    yield from yielded


def generate_solution_batches(
    row_names: list[str],
    bit_names: list[str],
    multi_matches: list[dict[str, Any]],
    batch_size: int = 500_000,
) -> Iterator[np.ndarray]:
    """Generate canonical graphs without materializing the complete universe."""
    rules_on_rows = [r for r in multi_matches if r["group"] == "rows"]
    rules_on_bits = [r for r in multi_matches if r["group"] == "bits"]
    if rules_on_rows:
        # Generate the transposed star forest, then convert it to fixed row masks.
        if rules_on_bits:
            raise DataValidationError(
                "A from-scratch model with unresolved multi-matches on both groups "
                "needs a dedicated generator; use an extend cast event for the observed case"
            )
        translated = [
            {"group": "bits", "size": r["size"], "known_members": r.get("known_members", [])}
            for r in rules_on_rows
        ]
        for transposed in generate_solution_batches(
            bit_names, row_names, translated, batch_size=batch_size
        ):
            converted = np.zeros((len(transposed), len(row_names)), dtype=mask_dtype(len(bit_names)))
            for bit_index in range(len(bit_names)):
                for row_index in range(len(row_names)):
                    present = (transposed[:, bit_index] & (1 << row_index)) != 0
                    converted[present, row_index] |= np.array(1 << bit_index, dtype=converted.dtype)
            yield converted
        return

    expected_groups = len(row_names)
    partitions = _multi_match_partitions(bit_names, rules_on_bits, expected_groups)
    permutations = faster_permutations(expected_groups)
    dtype = mask_dtype(len(bit_names))
    for partition in partitions:
        group_masks = np.array(
            [sum(1 << bit for bit in group) for group in partition], dtype=dtype
        )
        for start in range(0, len(permutations), batch_size):
            permutation_batch = permutations[start : start + batch_size]
            batch = np.zeros((len(permutation_batch), expected_groups), dtype=dtype)
            row_selector = np.arange(len(permutation_batch))
            for group_index, group_mask in enumerate(group_masks):
                batch[row_selector, permutation_batch[:, group_index]] = group_mask
            yield batch


def _event_pair_indices(state: SolverState, pair: list[str]) -> tuple[int, int]:
    if len(pair) != 2:
        raise DataValidationError(f"Expected a pair, got {pair!r}")
    if pair[0] in state.row_names and pair[1] in state.bit_names:
        return state.row_names.index(pair[0]), state.bit_names.index(pair[1])
    if pair[1] in state.row_names and pair[0] in state.bit_names:
        return state.row_names.index(pair[1]), state.bit_names.index(pair[0])
    raise DataValidationError(f"Pair must cross the two groups: {pair!r}")


def edge_mask(state: SolverState, solutions: np.ndarray, pair: list[str]) -> np.ndarray:
    row_index, bit_index = _event_pair_indices(state, pair)
    return (solutions[:, row_index] & np.array(1 << bit_index, dtype=solutions.dtype)) != 0


def _complete_pair_mask(
    state: SolverState, solutions: np.ndarray, pair: list[str]
) -> np.ndarray:
    """Require a normal one-to-one match, excluding either person from a multi-match."""
    row_index, bit_index = _event_pair_indices(state, pair)
    result = solutions[:, row_index] == np.array(1 << bit_index, dtype=solutions.dtype)
    bit_degree = np.zeros(len(solutions), dtype=np.uint8)
    bit = np.array(1 << bit_index, dtype=solutions.dtype)
    for other_row in range(len(state.row_names)):
        bit_degree += (solutions[:, other_row] & bit) != 0
    return result & (bit_degree == 1)


def matching_night_scores(
    state: SolverState, solutions: np.ndarray, event: dict[str, Any]
) -> np.ndarray:
    scores = np.full(len(solutions), int(event.get("automatic_lights", 0)), dtype=np.uint8)
    for pair in event["pairs"]:
        scores += edge_mask(state, solutions, pair)
    return scores


def _complete_group_mask(
    state: SolverState, solutions: np.ndarray, group: dict[str, Any]
) -> np.ndarray:
    partner = group["partner"]
    members = group["members"]
    result = np.ones(len(solutions), dtype=bool)
    for member in members:
        result &= edge_mask(state, solutions, [partner, member])
    if group.get("complete", True):
        if partner in state.row_names:
            row_index = state.row_names.index(partner)
            expected = 0
            for member in members:
                expected |= 1 << state.bit_names.index(member)
            result &= solutions[:, row_index] == expected
        else:
            bit_index = state.bit_names.index(partner)
            expected_rows = {state.row_names.index(member) for member in members}
            for row_index in range(len(state.row_names)):
                present = (solutions[:, row_index] & (1 << bit_index)) != 0
                result &= present if row_index in expected_rows else ~present
    return result


def _multi_match_member_mask(
    state: SolverState, solutions: np.ndarray, person: str
) -> np.ndarray:
    if person in state.bit_names:
        bit_index = state.bit_names.index(person)
        result = np.zeros(len(solutions), dtype=bool)
        for row_index in range(len(state.row_names)):
            contains = (solutions[:, row_index] & (1 << bit_index)) != 0
            degree = np.bitwise_count(solutions[:, row_index]) > 1
            result |= contains & degree
        return result
    if person in state.row_names:
        row_index = state.row_names.index(person)
        result = np.zeros(len(solutions), dtype=bool)
        for bit_index in range(len(state.bit_names)):
            contains = (solutions[:, row_index] & (1 << bit_index)) != 0
            degree = np.zeros(len(solutions), dtype=np.uint8)
            for other_row in range(len(state.row_names)):
                degree += (solutions[:, other_row] & (1 << bit_index)) != 0
            result |= contains & (degree > 1)
        return result
    raise DataValidationError(f"Unknown participant {person!r}")


def apply_evidence_event(
    state: SolverState, solutions: np.ndarray, event: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply one information-bearing event and return survivors and pre-night histogram."""
    event_type = event["type"]
    keep = np.ones(len(solutions), dtype=bool)
    light_distribution = None
    if event_type == "box":
        result = event["result"]
        if result != "sold":
            is_match = edge_mask(state, solutions, event["pair"])
            keep &= is_match if result == "yes" else ~is_match
            if result == "yes" and event.get(
                "complete_pair", not bool(event.get("revealed_group"))
            ):
                keep &= _complete_pair_mask(state, solutions, event["pair"])
        if event.get("revealed_group"):
            keep &= _complete_group_mask(state, solutions, event["revealed_group"])
    elif event_type == "matching_night":
        if event.get("result") == "sold":
            return solutions, None
        scores = matching_night_scores(state, solutions, event)
        light_distribution = np.bincount(
            scores,
            minlength=len(event["pairs"]) + int(event.get("automatic_lights", 0)) + 1,
        )
        keep &= scores == int(event["lights"])
    elif event_type == "solution":
        expected = np.zeros(len(state.row_names), dtype=solutions.dtype)
        for pair in event["pairs"]:
            row_index, bit_index = _event_pair_indices(state, pair)
            expected[row_index] |= np.array(1 << bit_index, dtype=solutions.dtype)
        keep &= np.all(solutions == expected, axis=1)
    elif event_type == "reveal":
        for pair in event.get("pairs", []):
            keep &= edge_mask(state, solutions, pair)
        for group in event.get("complete_groups", []):
            keep &= _complete_group_mask(state, solutions, group)
        for person in event.get("multi_match_members", []):
            keep &= _multi_match_member_mask(state, solutions, person)
    elif event_type == "cast_change":
        for person in event.get("multi_match_members", []):
            keep &= _multi_match_member_mask(state, solutions, person)
    else:
        raise DataValidationError(f"Unknown event type {event_type!r}")
    return solutions[keep], light_distribution


def event_is_evidence(event: dict[str, Any]) -> bool:
    if event["type"] == "cast_change":
        return bool(event.get("multi_match_members"))
    if event["type"] in {"box", "matching_night"} and event.get("result") == "sold":
        return False
    return event["type"] in {"box", "matching_night", "solution", "reveal"}


def _probabilities(solutions: np.ndarray, n_bits: int) -> np.ndarray:
    accumulator = _SnapshotAccumulator(solutions.shape[1], n_bits)
    accumulator.add(solutions)
    return accumulator.probabilities()


def _expand_with_new_participants(
    state: SolverState,
    old_solutions: np.ndarray,
    event: dict[str, Any],
    old_row_names: list[str],
    old_bit_names: list[str],
) -> np.ndarray:
    additions = event.get("add", [])
    if len(additions) != 1:
        raise DataValidationError("extend currently expects exactly one added participant")
    addition = additions[0]
    group = addition["group"]
    update = event.get("match_update", {})
    new_multi_match = update.get("new_multi_match", {})
    expected_side = "bits" if group == state.bit_group else "rows"
    if new_multi_match.get("group") != expected_side:
        raise DataValidationError(
            f"An added {group!r} participant must extend a {expected_side!r} multi-match"
        )
    target_size = int(new_multi_match.get("size", 2))
    eligible = update.get("eligible_partners")
    if eligible is None:
        other_group = state.bit_group if group == state.row_group else state.row_group
        eligible = sorted(state.active[other_group], key=state.groups[other_group].index)

    expanded: list[np.ndarray] = []
    dtype = state.mask_dtype
    base = np.zeros((len(old_solutions), len(state.row_names)), dtype=dtype)
    for old_row_index, row_name in enumerate(old_row_names):
        new_row_index = state.row_names.index(row_name)
        old_values = old_solutions[:, old_row_index]
        for old_bit_index, bit_name in enumerate(old_bit_names):
            present = (old_values & np.array(1 << old_bit_index, dtype=old_solutions.dtype)) != 0
            base[present, new_row_index] |= np.array(
                1 << state.bit_names.index(bit_name), dtype=dtype
            )
    if group == state.bit_group:
        new_bit_index = state.bit_names.index(addition["person"])
        for partner in eligible:
            row_index = state.row_names.index(partner)
            valid = np.bitwise_count(base[:, row_index]) == target_size - 1
            candidate = base[valid].copy()
            candidate[:, row_index] |= np.array(1 << new_bit_index, dtype=dtype)
            expanded.append(candidate)
    elif group == state.row_group:
        new_row_index = state.row_names.index(addition["person"])
        for partner in eligible:
            bit_index = state.bit_names.index(partner)
            degree = np.zeros(len(base), dtype=np.uint8)
            for row_name in old_row_names:
                row_index = state.row_names.index(row_name)
                degree += (base[:, row_index] & (1 << bit_index)) != 0
            valid = degree == target_size - 1
            candidate = base[valid].copy()
            candidate[:, new_row_index] = np.array(1 << bit_index, dtype=dtype)
            expanded.append(candidate)
    else:
        raise DataValidationError(f"Unknown addition group {group!r}")
    if not expanded:
        raise NoSolutionsError("The cast extension produced no possible match graphs")
    return np.concatenate(expanded, axis=0)


def _snapshot_from_accumulator(
    state: SolverState,
    accumulator: _SnapshotAccumulator,
    event: dict[str, Any],
    week: int,
    event_number: int,
    event_index: int,
    before_count: int,
    light_distribution: np.ndarray | None = None,
) -> Snapshot:
    normalized_lights = None
    if light_distribution is not None:
        total = light_distribution.sum()
        normalized_lights = light_distribution.astype(np.float64) / total if total else light_distribution.astype(np.float64)
    return Snapshot(
        week=week,
        event_number=event_number,
        event_index=event_index,
        event_type=event["type"],
        title=event.get("title", default_event_title(event)),
        before_count=before_count,
        solution_count=accumulator.count,
        probabilities=accumulator.probabilities(),
        row_names=tuple(state.row_names),
        bit_names=tuple(state.bit_names),
        active={key: tuple(name for name in values if name in state.active[key]) for key, values in state.groups.items()},
        face_layer=state.face_layer,
        light_distribution=normalized_lights,
        actual_lights=event.get("lights"),
        combination_sample=accumulator.combination_sample(),
    )


def default_event_title(event: dict[str, Any]) -> str:
    return {
        "entry": "Einzug",
        "box": "Match Box",
        "matching_night": "Matching Night",
        "cast_change": "Cast Update",
        "solution": "Auflösung",
        "reveal": "Neue Information",
    }.get(event["type"], event["type"])


class SeasonSolver:
    """Process one season while retaining only canonical surviving graphs."""

    def __init__(self, season: str, data: dict[str, Any]) -> None:
        matrix = data["matrix"]
        groups = {key: list(value) for key, value in data["groups"].items()}
        self.data = data
        initial_active = data["initial_active"]
        if initial_active == "all":
            active = {key: set(value) for key, value in groups.items()}
        else:
            active = {key: set(initial_active[key]) for key in groups}
        self.state = SolverState(
            season=season,
            row_group=matrix["rows"],
            bit_group=matrix["bits"],
            groups=groups,
            active=active,
            multi_matches=[dict(rule) for rule in data["initial_match_model"].get("multi_matches", [])],
            face_layer=data["style"]["face_layer"],
        )
        self.snapshots: list[Snapshot] = []

    def _apply_cast_state(self, event: dict[str, Any]) -> tuple[list[str], list[str]]:
        old_rows = list(self.state.row_names)
        old_bits = list(self.state.bit_names)
        for addition in event.get("add", []):
            group = addition["group"]
            person = addition["person"]
            self.state.groups[group].append(person)
            self.state.groups[group].sort(key=str.casefold)
            self.state.active[group].add(person)
        for person in event.get("remove", []):
            for group in self.state.groups:
                self.state.active[group].discard(person)
        if event.get("face_layer"):
            self.state.face_layer = event["face_layer"]
        update = event.get("match_update", {})
        if update.get("mode") == "rebuild":
            self.state.multi_matches = [dict(rule) for rule in update.get("multi_matches", [])]
        elif update.get("mode") == "extend" and update.get("new_multi_match"):
            self.state.multi_matches.append(dict(update["new_multi_match"]))
        return old_rows, old_bits

    def _all_prior_evidence(self, until_event_index: int) -> list[dict[str, Any]]:
        events = flatten_events(self.data)
        return [event for _, _, event in events[:until_event_index] if event_is_evidence(event)]

    def _stream_segment(
        self,
        events: list[tuple[int, int, int, dict[str, Any]]],
        replay: list[dict[str, Any]],
        include_entry: bool = False,
        leading_cast: tuple[int, int, int, dict[str, Any]] | None = None,
    ) -> None:
        stages: list[_SnapshotAccumulator] = []
        stage_events: list[tuple[int, int, int, dict[str, Any]]] = []
        if include_entry:
            stages.append(_SnapshotAccumulator(len(self.state.row_names), len(self.state.bit_names)))
            stage_events.append((1, 0, -1, {"type": "entry", "title": "Einzug"}))
        if leading_cast:
            stages.append(_SnapshotAccumulator(len(self.state.row_names), len(self.state.bit_names)))
            stage_events.append(leading_cast)
        for item in events:
            stages.append(_SnapshotAccumulator(len(self.state.row_names), len(self.state.bit_names)))
            stage_events.append(item)
        light_totals: list[np.ndarray | None] = [None] * len(stages)
        before_totals = [0] * len(stages)
        survivors: list[np.ndarray] = []

        for batch in generate_solution_batches(
            self.state.row_names, self.state.bit_names, self.state.multi_matches
        ):
            current = batch
            for old_event in replay:
                current, _ = apply_evidence_event(self.state, current, old_event)
                if len(current) == 0:
                    break
            stage_index = 0
            if include_entry:
                # Entry is meaningful only before evidence. For a rebuild segment it is false.
                before_totals[stage_index] += len(batch)
                stages[stage_index].add(batch)
                stage_index += 1
            if leading_cast:
                before_totals[stage_index] += len(current)
                stages[stage_index].add(current)
                stage_index += 1
            for _, _, _, event in events:
                before_count = len(current)
                before_totals[stage_index] += before_count
                current, lights = apply_evidence_event(self.state, current, event)
                stages[stage_index].add(current)
                if lights is not None:
                    if light_totals[stage_index] is None:
                        light_totals[stage_index] = np.zeros(len(lights), dtype=np.uint64)
                    elif len(light_totals[stage_index]) < len(lights):
                        light_totals[stage_index] = np.pad(light_totals[stage_index], (0, len(lights) - len(light_totals[stage_index])))
                    light_totals[stage_index][: len(lights)] += lights.astype(np.uint64)
                stage_index += 1
                if before_count and len(current) == 0:
                    break
            if len(current):
                survivors.append(current)

        for index, (week, event_number, event_index, event) in enumerate(stage_events):
            snapshot = _snapshot_from_accumulator(
                self.state,
                stages[index],
                event,
                week,
                event_number,
                event_index,
                before_totals[index],
                light_totals[index],
            )
            self.snapshots.append(snapshot)
            self.state.event_records.append(
                EventCountRecord(
                    week,
                    event_number,
                    event_index,
                    event["type"],
                    snapshot.before_count,
                    snapshot.solution_count,
                )
            )
            if snapshot.light_distribution is not None:
                self.state.night_records.append(
                    NightRecord(week, event_number, snapshot.light_distribution, int(event["lights"]))
                )
        if not survivors:
            locator = stage_events[-1][:3] if stage_events else (0, 0, 0)
            raise NoSolutionsError(f"No solutions remain after week/event {locator[:2]}")
        self.state.solutions = np.concatenate(survivors, axis=0)

    def _process_in_memory_event(
        self, week: int, event_number: int, event_index: int, event: dict[str, Any]
    ) -> None:
        assert self.state.solutions is not None
        before = self.state.solutions
        filtered, lights = apply_evidence_event(self.state, before, event)
        if len(filtered) == 0:
            raise NoSolutionsError(
                f"Event {week}-{event_number} ({event['type']}) removes every solution"
            )
        self.state.solutions = filtered
        accumulator = _SnapshotAccumulator(len(self.state.row_names), len(self.state.bit_names))
        accumulator.add(filtered)
        light_distribution = None
        if lights is not None:
            light_distribution = lights
        snapshot = _snapshot_from_accumulator(
            self.state,
            accumulator,
            event,
            week,
            event_number,
            event_index,
            len(before),
            light_distribution,
        )
        self.snapshots.append(snapshot)
        self.state.event_records.append(
            EventCountRecord(
                week,
                event_number,
                event_index,
                event["type"],
                snapshot.before_count,
                snapshot.solution_count,
            )
        )
        if snapshot.light_distribution is not None:
            self.state.night_records.append(
                NightRecord(week, event_number, snapshot.light_distribution, int(event["lights"]))
            )

    def process_week(
        self,
        week: dict[str, Any],
        start_global_index: int,
        event_number_offset: int = 0,
    ) -> None:
        numbered = [
            (
                int(week["number"]),
                event_number_offset + number,
                start_global_index + number - 1,
                event,
            )
            for number, event in enumerate(week["events"], start=1)
        ]
        cursor = 0
        if self.state.solutions is None:
            structural = next(
                (
                    i
                    for i, (_, _, _, e) in enumerate(numbered)
                    if e["type"] == "cast_change" and e.get("match_update", {}).get("mode") in {"rebuild", "extend"}
                ),
                None,
            )
            prefix = numbered if structural is None else numbered[:structural]
            self._stream_segment(prefix, replay=[], include_entry=self.state.processed_events == 0)
            cursor = len(prefix)

        while cursor < len(numbered):
            week_no, event_no, event_index, event = numbered[cursor]
            if event["type"] == "cast_change":
                old_solution_count = len(self.state.solutions) if self.state.solutions is not None else 0
                old_rows, old_bits = self._apply_cast_state(event)
                mode = event.get("match_update", {}).get("mode")
                if mode == "rebuild":
                    replay = self._all_prior_evidence(event_index)
                    following: list[tuple[int, int, int, dict[str, Any]]] = []
                    j = cursor + 1
                    while j < len(numbered):
                        candidate = numbered[j]
                        if candidate[3]["type"] == "cast_change" and candidate[3].get("match_update", {}).get("mode") in {"rebuild", "extend"}:
                            break
                        following.append(candidate)
                        j += 1
                    self._stream_segment(
                        following,
                        replay=replay,
                        leading_cast=(week_no, event_no, event_index, event),
                    )
                    cursor = j
                    continue
                if mode == "extend":
                    assert self.state.solutions is not None
                    self.state.solutions = _expand_with_new_participants(
                        self.state, self.state.solutions, event, old_rows, old_bits
                    )
                if event.get("multi_match_members"):
                    self._process_in_memory_event(week_no, event_no, event_index, event)
                else:
                    accumulator = _SnapshotAccumulator(len(self.state.row_names), len(self.state.bit_names))
                    accumulator.add(self.state.solutions)
                    self.snapshots.append(
                        _snapshot_from_accumulator(
                            self.state,
                            accumulator,
                            event,
                            week_no,
                            event_no,
                            event_index,
                            old_solution_count,
                        )
                    )
                    self.state.event_records.append(
                        EventCountRecord(
                            week_no,
                            event_no,
                            event_index,
                            event["type"],
                            old_solution_count,
                            len(self.state.solutions),
                        )
                    )
            else:
                self._process_in_memory_event(week_no, event_no, event_index, event)
            cursor += 1

        self.state.processed_events = start_global_index + len(numbered)
        self.state.completed_week = int(week["number"])


def flatten_events(data: dict[str, Any]) -> list[tuple[int, int, dict[str, Any]]]:
    return [
        (int(week["number"]), event_number, event)
        for week in data["weeks"]
        for event_number, event in enumerate(week["events"], start=1)
    ]


def event_prefix_hash(data: dict[str, Any], event_count: int) -> str:
    payload = {
        "solver_version": SOLVER_VERSION,
        "schema_version": data.get("schema_version", SCHEMA_VERSION),
        "groups": data["groups"],
        "initial_active": data["initial_active"],
        "matrix": data["matrix"],
        "initial_match_model": data["initial_match_model"],
        "events": [event for _, _, event in flatten_events(data)[:event_count]],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_match_model(
    where: str,
    row_names: list[str],
    bit_names: list[str],
    model: dict[str, Any],
) -> None:
    rules = model.get("multi_matches", [])
    sides = {rule.get("group") for rule in rules}
    if not sides <= {"rows", "bits"}:
        raise DataValidationError(f"{where}: multi-match group must be 'rows' or 'bits'")
    if len(sides) > 1:
        raise DataValidationError(
            f"{where}: unresolved multi-matches on both sides need a dedicated generator"
        )

    names_by_side = {"rows": row_names, "bits": bit_names}
    used_known: set[str] = set()
    for rule in rules:
        side = rule["group"]
        size = int(rule.get("size", 0))
        known = list(rule.get("known_members", []))
        if size < 2:
            raise DataValidationError(f"{where}: multi-match size must be at least two")
        if size > len(names_by_side[side]):
            raise DataValidationError(f"{where}: multi-match is larger than its group")
        if len(known) != len(set(known)) or len(known) > size:
            raise DataValidationError(f"{where}: invalid known multi-match members")
        if any(name not in names_by_side[side] for name in known):
            raise DataValidationError(f"{where}: known member is on the wrong side")
        if used_known.intersection(known):
            raise DataValidationError(f"{where}: known member occurs in multiple multi-matches")
        used_known.update(known)

    row_slots = len(row_names) - sum(
        int(rule["size"]) - 1 for rule in rules if rule["group"] == "rows"
    )
    bit_slots = len(bit_names) - sum(
        int(rule["size"]) - 1 for rule in rules if rule["group"] == "bits"
    )
    if row_slots != bit_slots:
        raise DataValidationError(
            f"{where}: cast sizes and multi-match structure are incompatible "
            f"({row_slots} row slots versus {bit_slots} bit slots)"
        )


def validate_season_data(season: str, data: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "groups",
        "initial_active",
        "matrix",
        "initial_match_model",
        "style",
        "weeks",
    }
    missing = required - data.keys()
    if missing:
        raise DataValidationError(f"{season}: missing fields {sorted(missing)}")
    if int(data["schema_version"]) != SCHEMA_VERSION:
        raise DataValidationError(f"{season}: expected schema version {SCHEMA_VERSION}")
    status = data.get("status", "ongoing")
    outcome = data.get("outcome")
    competitive_end_week = data.get("competitive_end_week")
    target_lights = data.get("target_lights", 10)
    if status not in {"completed", "ongoing"}:
        raise DataValidationError(f"{season}: status must be completed or ongoing")
    if outcome not in {"won", "lost", None}:
        raise DataValidationError(f"{season}: outcome must be won, lost, or null")
    if status == "completed":
        if outcome not in {"won", "lost"} or not isinstance(competitive_end_week, int):
            raise DataValidationError(
                f"{season}: completed seasons need an outcome and competitive_end_week"
            )
    elif outcome is not None or competitive_end_week is not None:
        raise DataValidationError(
            f"{season}: ongoing seasons need null outcome and competitive_end_week"
        )
    if not isinstance(target_lights, int) or target_lights < 1:
        raise DataValidationError(f"{season}: target_lights must be a positive integer")
    if len(data["groups"]) != 2:
        raise DataValidationError(f"{season}: exactly two participant groups are required")
    row_group = data["matrix"]["rows"]
    bit_group = data["matrix"]["bits"]
    if row_group == bit_group or {row_group, bit_group} != set(data["groups"]):
        raise DataValidationError(f"{season}: matrix rows/bits must reference both groups")
    all_names = [name for names in data["groups"].values() for name in names]
    if len(all_names) != len(set(all_names)):
        raise DataValidationError(f"{season}: participant names must be globally unique")
    current = {key: list(value) for key, value in data["groups"].items()}
    if data["initial_active"] == "all":
        active = {key: set(value) for key, value in current.items()}
    elif isinstance(data["initial_active"], dict):
        if set(data["initial_active"]) != set(current):
            raise DataValidationError(f"{season}: initial_active must contain both groups")
        active = {key: set(data["initial_active"][key]) for key in current}
        for key in current:
            if not active[key] <= set(current[key]):
                raise DataValidationError(f"{season}: initial_active contains an unknown participant")
    else:
        raise DataValidationError(f"{season}: initial_active must be 'all' or a group mapping")
    _validate_match_model(
        f"{season} initial model",
        current[row_group],
        current[bit_group],
        data["initial_match_model"],
    )
    previous_week = 0
    for week in data["weeks"]:
        number = int(week["number"])
        if number <= previous_week:
            raise DataValidationError(f"{season}: week numbers must increase")
        previous_week = number
        for event_number, event in enumerate(week["events"], start=1):
            where = f"{season} week {number} event {event_number}"
            event_type = event.get("type")
            if event_type not in {
                "box",
                "matching_night",
                "cast_change",
                "solution",
                "reveal",
            }:
                raise DataValidationError(f"{where}: unknown event type {event_type!r}")
            if event["type"] == "cast_change":
                additions = event.get("add", [])
                for addition in event.get("add", []):
                    if addition["group"] not in current:
                        raise DataValidationError(f"{where}: unknown group {addition['group']!r}")
                    if addition["person"] in {n for values in current.values() for n in values}:
                        raise DataValidationError(f"{where}: duplicate participant {addition['person']!r}")
                    current[addition["group"]].append(addition["person"])
                    current[addition["group"]].sort(key=str.casefold)
                    active[addition["group"]].add(addition["person"])
                for person in event.get("remove", []):
                    if person not in {n for values in current.values() for n in values}:
                        raise DataValidationError(f"{where}: unknown removed participant {person!r}")
                    for names in active.values():
                        names.discard(person)
                update = event.get("match_update", {})
                mode = update.get("mode")
                if additions and mode not in {"rebuild", "extend"}:
                    raise DataValidationError(
                        f"{where}: an arrival requires match_update mode rebuild or extend"
                    )
                if mode == "rebuild":
                    _validate_match_model(
                        f"{where} rebuild",
                        current[row_group],
                        current[bit_group],
                        {"multi_matches": update.get("multi_matches", [])},
                    )
                elif mode == "extend":
                    if len(additions) != 1 or "new_multi_match" not in update:
                        raise DataValidationError(
                            f"{where}: extend requires one arrival and new_multi_match"
                        )
                    addition = additions[0]
                    expected_side = "bits" if addition["group"] == bit_group else "rows"
                    rule = update["new_multi_match"]
                    if rule.get("group") != expected_side:
                        raise DataValidationError(
                            f"{where}: extended multi-match is on the wrong side"
                        )
                    if addition["person"] not in rule.get("known_members", []):
                        raise DataValidationError(
                            f"{where}: the arriving participant must be a known member"
                        )
                    if int(rule.get("size", 0)) < 2:
                        raise DataValidationError(f"{where}: invalid extended multi-match size")
                    other_group = bit_group if addition["group"] == row_group else row_group
                    eligible = update.get("eligible_partners", active[other_group])
                    if any(person not in active[other_group] for person in eligible):
                        raise DataValidationError(
                            f"{where}: extend has an unknown or inactive eligible partner"
                        )
            known = {n for values in current.values() for n in values}
            pairs: Iterable[list[str]] = []
            if event["type"] == "box":
                if event.get("result") not in {"yes", "no", "sold"}:
                    raise DataValidationError(f"{where}: invalid Match Box result")
                if event.get("complete_pair") and event.get("result") != "yes":
                    raise DataValidationError(
                        f"{where}: complete_pair is only valid for a yes Match Box"
                    )
                pairs = [event["pair"]]
            elif event["type"] == "matching_night":
                pairs = event["pairs"]
                result = event.get("result", "normal")
                if result not in {"normal", "sold"}:
                    raise DataValidationError(f"{where}: invalid Matching Night result")
                if result == "sold":
                    if "lights" in event:
                        raise DataValidationError(
                            f"{where}: a sold Matching Night must not contain lights"
                        )
                else:
                    if "lights" not in event:
                        raise DataValidationError(f"{where}: Matching Night lights are required")
                    automatic = int(event.get("automatic_lights", 0))
                    maximum = len(event["pairs"]) + automatic
                    if not automatic <= int(event["lights"]) <= maximum:
                        raise DataValidationError(f"{where}: impossible light count")
            elif event["type"] in {"solution", "reveal"}:
                pairs = event.get("pairs", [])
            for pair in pairs:
                if len(pair) != 2 or any(person not in known for person in pair):
                    raise DataValidationError(f"{where}: invalid pair {pair!r}")
                first_group = next(key for key, names in current.items() if pair[0] in names)
                second_group = next(key for key, names in current.items() if pair[1] in names)
                if first_group == second_group:
                    raise DataValidationError(f"{where}: pair does not cross groups: {pair!r}")
            if event["type"] == "solution":
                covered = {person for pair in event["pairs"] for person in pair}
                if covered != known:
                    raise DataValidationError(
                        f"{where}: a solution must list every revealed participant"
                    )
            if event["type"] == "matching_night":
                seated = [person for pair in event["pairs"] for person in pair]
                if len(seated) != len(set(seated)):
                    raise DataValidationError(f"{where}: a participant is seated more than once")
                if any(person not in {n for values in active.values() for n in values} for person in seated):
                    raise DataValidationError(f"{where}: an inactive participant is seated")

            revealed_groups: list[dict[str, Any]] = []
            if event.get("revealed_group"):
                revealed_groups.append(event["revealed_group"])
            revealed_groups.extend(event.get("complete_groups", []))
            for revealed in revealed_groups:
                people = [revealed.get("partner"), *revealed.get("members", [])]
                if len(people) < 3 or any(person not in known for person in people):
                    raise DataValidationError(f"{where}: invalid complete multi-match reveal")
                partner_group = next(key for key, names in current.items() if people[0] in names)
                if any(
                    next(key for key, names in current.items() if member in names) == partner_group
                    for member in people[1:]
                ):
                    raise DataValidationError(f"{where}: multi-match reveal does not cross groups")

            for person in event.get("multi_match_members", []):
                if person not in known:
                    raise DataValidationError(f"{where}: unknown multi-match member {person!r}")


def serialize_night_records(records: list[NightRecord]) -> list[dict[str, Any]]:
    return [
        {
            "week": record.week,
            "event_number": record.event_number,
            "distribution": record.distribution.tolist(),
            "actual_lights": record.actual_lights,
        }
        for record in records
    ]


def deserialize_night_records(records: list[dict[str, Any]]) -> list[NightRecord]:
    return [
        NightRecord(
            int(record["week"]),
            int(record["event_number"]),
            np.asarray(record["distribution"], dtype=np.float64),
            int(record["actual_lights"]),
        )
        for record in records
    ]


def serialize_event_records(records: list[EventCountRecord]) -> list[dict[str, Any]]:
    return [
        {
            "week": record.week,
            "event_number": record.event_number,
            "event_index": record.event_index,
            "event_type": record.event_type,
            "before_count": record.before_count,
            "after_count": record.after_count,
        }
        for record in records
    ]


def deserialize_event_records(records: list[dict[str, Any]]) -> list[EventCountRecord]:
    return [
        EventCountRecord(
            week=int(record["week"]),
            event_number=int(record["event_number"]),
            event_index=int(record["event_index"]),
            event_type=str(record["event_type"]),
            before_count=int(record["before_count"]),
            after_count=int(record["after_count"]),
        )
        for record in records
    ]
