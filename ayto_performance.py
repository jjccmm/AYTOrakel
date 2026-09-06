"""Historical season metrics and transparent win comparisons."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ayto_solver import EventCountRecord


METRIC_VERSION = 2
# Only bump this when solver semantics change historical counts or metrics.
# Checkpoint-only representation changes may use a newer SOLVER_VERSION.
HISTORY_SOLVER_VERSION = 3
HISTORY_FILE = Path("historical_metrics.csv")


class HistoryError(ValueError):
    """Raised when the historical dataset is missing, stale, or malformed."""


@dataclass(frozen=True)
class PerformanceRecord:
    season: str
    week: int
    status: str
    outcome: str
    competitive_end_week: int | None
    target_lights: int
    source_hash: str
    event_count: int
    solutions_remaining: int
    information_bits: float
    information_progress: float
    decision_count: int
    mean_elimination: float
    observed_nights: int
    latest_lights: int | None
    week_lights: int | None
    cumulative_lights: int
    light_form: float
    secure_matches: int
    confirmed_boxes: int
    double_match_found: bool
    box_information_gain: float
    night_information_gain: float
    terminal: bool
    metric_version: int = METRIC_VERSION

@dataclass(frozen=True)
class MetricComparison:
    key: str
    label: str
    probability: float
    comparable_seasons: int


@dataclass(frozen=True)
class WinPrediction:
    probability: float
    low: float
    high: float
    completed_seasons: int
    standard_deviation: float
    comparisons: tuple[MetricComparison, ...]


def season_source_hash(data: dict[str, Any]) -> str:
    payload = {
        "metric_version": METRIC_VERSION,
        "solver_version": HISTORY_SOLVER_VERSION,
        "schema_version": data.get("schema_version"),
        "status": data.get("status"),
        "outcome": data.get("outcome"),
        "competitive_end_week": data.get("competitive_end_week"),
        "target_lights": data.get("target_lights", 10),
        "groups": data.get("groups"),
        "initial_active": data.get("initial_active"),
        "matrix": data.get("matrix"),
        "initial_match_model": data.get("initial_match_model"),
        "weeks": data.get("weeks"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _event_information(event: dict[str, Any], before: int, after: int) -> float:
    if before <= 0 or after <= 0 or after >= before:
        return 0.0
    if event["type"] in {"box", "matching_night"} and event.get("result") == "sold":
        return 0.0
    if event["type"] == "cast_change" and not event.get("multi_match_members"):
        return 0.0
    if event["type"] not in {"box", "matching_night", "cast_change", "reveal", "solution"}:
        return 0.0
    return max(0.0, math.log2(before / after))


def _maximum_matching(
    edges: set[tuple[str, str]], active_rows: set[str], active_bits: set[str]
) -> int:
    adjacency: dict[str, list[str]] = {name: [] for name in active_rows}
    for row, bit in edges:
        if row in active_rows and bit in active_bits:
            adjacency[row].append(bit)
    matched: dict[str, str] = {}

    def augment(row: str, seen: set[str]) -> bool:
        for bit in adjacency[row]:
            if bit in seen:
                continue
            seen.add(bit)
            if bit not in matched or augment(matched[bit], seen):
                matched[bit] = row
                return True
        return False

    return sum(augment(row, set()) for row in sorted(active_rows))


def _canonical_edge(
    pair: Sequence[str], row_names: set[str], bit_names: set[str]
) -> tuple[str, str]:
    first, second = pair
    if first in row_names and second in bit_names:
        return first, second
    if second in row_names and first in bit_names:
        return second, first
    raise HistoryError(f"Pair does not cross the configured groups: {list(pair)!r}")


def build_performance_records(
    season: str,
    data: dict[str, Any],
    event_records: Sequence[EventCountRecord],
) -> list[PerformanceRecord]:
    """Reduce exact solver counts and public events to one row per week."""
    record_by_index = {record.event_index: record for record in event_records}
    entry = record_by_index.get(-1)
    if entry is None:
        raise HistoryError(f"{season}: solver history has no initial-universe record")

    row_group = data["matrix"]["rows"]
    bit_group = data["matrix"]["bits"]
    groups = {key: list(names) for key, names in data["groups"].items()}
    if data["initial_active"] == "all":
        active = {key: set(names) for key, names in groups.items()}
    else:
        active = {key: set(data["initial_active"][key]) for key in groups}
    row_names = set(groups[row_group])
    bit_names = set(groups[bit_group])
    confirmed_edges: set[tuple[str, str]] = set()

    cumulative_bits = 0.0
    decisions = 0
    light_form = 0.0
    observed_nights = 0
    latest_lights: int | None = None
    cumulative_lights = 0
    confirmed_boxes = 0
    double_match_found = False
    source_hash = season_source_hash(data)
    event_count = 0
    current_count = entry.after_count
    output: list[PerformanceRecord] = []

    for week in data["weeks"]:
        week_number = int(week["number"])
        if event_count not in record_by_index:
            break
        week_lights: int | None = None
        week_box_bits = 0.0
        week_night_bits = 0.0
        for event in week["events"]:
            count_record = record_by_index.get(event_count)
            if count_record is None:
                raise HistoryError(
                    f"{season}: solver history ends before week {week_number}, event "
                    f"{event_count + 1}"
                )
            current_count = count_record.after_count
            event_bits = _event_information(
                event, count_record.before_count, count_record.after_count
            )
            cumulative_bits += event_bits
            if event["type"] == "box":
                week_box_bits += event_bits
            elif event["type"] == "matching_night":
                week_night_bits += event_bits
            if event["type"] in {"box", "matching_night"}:
                decisions += 1
            if event["type"] == "box" and event.get("result") == "yes":
                confirmed_boxes += 1
            if event["type"] == "matching_night" and event.get("result") != "sold":
                latest_lights = int(event["lights"])
                week_lights = latest_lights
                cumulative_lights += latest_lights
                quote = latest_lights / int(data["target_lights"])
                light_form = quote if observed_nights == 0 else 0.5 * quote + 0.5 * light_form
                observed_nights += 1

            if event["type"] == "cast_change":
                for addition in event.get("add", []):
                    person = addition["person"]
                    group = addition["group"]
                    groups[group].append(person)
                    groups[group].sort(key=str.casefold)
                    active[group].add(person)
                    (row_names if group == row_group else bit_names).add(person)
                for person in event.get("remove", []):
                    for names in active.values():
                        names.discard(person)

            pairs: list[Sequence[str]] = []
            if event["type"] == "box" and event.get("result") == "yes":
                pairs.append(event["pair"])
            elif event["type"] in {"solution", "reveal"}:
                pairs.extend(event.get("pairs", []))
            if event.get("revealed_group"):
                group = event["revealed_group"]
                pairs.extend([[group["partner"], member] for member in group["members"]])
                if len(group.get("members", [])) > 1:
                    double_match_found = True
            for group in event.get("complete_groups", []):
                pairs.extend([[group["partner"], member] for member in group["members"]])
                if len(group.get("members", [])) > 1:
                    double_match_found = True
            for pair in pairs:
                confirmed_edges.add(_canonical_edge(pair, row_names, bit_names))
            event_count += 1

        remaining_bits = math.log2(current_count) if current_count > 1 else 0.0
        denominator = cumulative_bits + remaining_bits
        progress = cumulative_bits / denominator if denominator else 1.0
        elimination = 1.0 - 2.0 ** (-cumulative_bits / decisions) if decisions else 0.0
        competitive_end = data["competitive_end_week"]
        output.append(
            PerformanceRecord(
                season=season,
                week=week_number,
                status=data["status"],
                outcome=data["outcome"] or "",
                competitive_end_week=competitive_end,
                target_lights=int(data["target_lights"]),
                source_hash=source_hash,
                event_count=event_count,
                solutions_remaining=current_count,
                information_bits=cumulative_bits,
                information_progress=progress,
                decision_count=decisions,
                mean_elimination=elimination,
                observed_nights=observed_nights,
                latest_lights=latest_lights,
                week_lights=week_lights,
                cumulative_lights=cumulative_lights,
                light_form=light_form,
                secure_matches=_maximum_matching(
                    confirmed_edges, row_names, bit_names
                ),
                confirmed_boxes=confirmed_boxes,
                double_match_found=double_match_found,
                box_information_gain=1.0 - 2.0 ** (-week_box_bits),
                night_information_gain=1.0 - 2.0 ** (-week_night_bits),
                terminal=bool(
                    data["status"] == "completed"
                    and competitive_end is not None
                    and week_number >= int(competitive_end)
                ),
            )
        )
    processed_count = sum(record.event_index >= 0 for record in event_records)
    if event_count != processed_count:
        raise HistoryError(f"{season}: solver history stops inside a configured week")
    return output


def write_history(records: Iterable[PerformanceRecord], path: Path = HISTORY_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    names = [field.name for field in fields(PerformanceRecord)]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["competitive_end_week"] = record.competitive_end_week or ""
            row["latest_lights"] = "" if record.latest_lights is None else record.latest_lights
            writer.writerow(row)
    os.replace(temporary, path)


def read_history(path: Path = HISTORY_FILE) -> list[PerformanceRecord]:
    if not path.is_file():
        raise HistoryError(f"Historical dataset not found: {path}")
    records: list[PerformanceRecord] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                records.append(
                    PerformanceRecord(
                        season=row["season"], week=int(row["week"]), status=row["status"],
                        outcome=row["outcome"],
                        competitive_end_week=int(row["competitive_end_week"]) if row["competitive_end_week"] else None,
                        target_lights=int(row["target_lights"]), source_hash=row["source_hash"],
                        event_count=int(row["event_count"]), solutions_remaining=int(row["solutions_remaining"]),
                        information_bits=float(row["information_bits"]),
                        information_progress=float(row["information_progress"]),
                        decision_count=int(row["decision_count"]), mean_elimination=float(row["mean_elimination"]),
                        observed_nights=int(row["observed_nights"]),
                        latest_lights=int(row["latest_lights"]) if row["latest_lights"] else None,
                        week_lights=int(row["week_lights"]) if row["week_lights"] else None,
                        cumulative_lights=int(row["cumulative_lights"]),
                        light_form=float(row["light_form"]), secure_matches=int(row["secure_matches"]),
                        confirmed_boxes=int(row["confirmed_boxes"]),
                        double_match_found=row["double_match_found"].lower() == "true",
                        box_information_gain=float(row["box_information_gain"]),
                        night_information_gain=float(row["night_information_gain"]),
                        terminal=row["terminal"].lower() == "true", metric_version=int(row["metric_version"]),
                    )
                )
    except (KeyError, TypeError, ValueError) as error:
        raise HistoryError(f"Historical dataset is malformed: {error}") from error
    return records


def validate_history(
    records: Sequence[PerformanceRecord], seasons: dict[str, dict[str, Any]]
) -> None:
    by_season: dict[str, list[PerformanceRecord]] = {}
    for record in records:
        by_season.setdefault(record.season, []).append(record)
    problems: list[str] = []
    for season, data in seasons.items():
        rows = by_season.get(season, [])
        expected_weeks = [int(week["number"]) for week in data["weeks"]]
        if [row.week for row in rows] != expected_weeks:
            problems.append(f"{season}: Wochen fehlen oder sind veraltet")
            continue
        expected_hash = season_source_hash(data)
        if any(row.source_hash != expected_hash or row.metric_version != METRIC_VERSION for row in rows):
            problems.append(f"{season}: Quelldaten wurden geändert")
    extras = set(by_season) - set(seasons)
    if extras:
        problems.append("unbekannte Staffeln: " + ", ".join(sorted(extras)))
    if problems:
        raise HistoryError("Historisches Dataset ist veraltet (" + "; ".join(problems) + ")")


def _completed_night_rows(
    records: Sequence[PerformanceRecord], night_number: int
) -> dict[str, PerformanceRecord]:
    rows: dict[str, PerformanceRecord] = {}
    for row in records:
        if (
            row.status == "completed"
            and row.outcome in {"won", "lost"}
            and row.week_lights is not None
            and row.observed_nights == night_number
        ):
            rows[row.season] = row
    return rows


def predict_win_chance(
    records: Sequence[PerformanceRecord], target: PerformanceRecord
) -> WinPrediction:
    """Compare the current state with equally far, same-or-worse old seasons."""
    outcomes = {
        row.season: row.outcome
        for row in records
        if row.status == "completed" and row.outcome in {"won", "lost"}
    }
    if not outcomes:
        raise HistoryError("No completed seasons are available for comparison")
    night_rows = _completed_night_rows(records, target.observed_nights)
    if not night_rows or target.latest_lights is None:
        raise HistoryError(
            f"No historical seasons have a comparable night {target.observed_nights}"
        )

    definitions = (
        ("night_lights", "Lichter dieser Nacht", lambda row: row.week_lights, target.latest_lights),
        ("total_lights", "Lichter insgesamt", lambda row: row.cumulative_lights, target.cumulative_lights),
        ("confirmed_boxes", "Bestätigte Match Boxes", lambda row: row.confirmed_boxes, target.confirmed_boxes),
        ("progress", "Rätsel gelöst", lambda row: row.information_progress, target.information_progress),
    )
    comparisons: list[MetricComparison] = []
    for key, label, getter, threshold in definitions:
        comparable = [
            season
            for season, row in night_rows.items()
            if getter(row) is not None and getter(row) <= threshold
        ]
        if comparable:
            probability = sum(outcomes[season] == "won" for season in comparable) / len(comparable)
        else:
            probability = sum(outcome == "won" for outcome in outcomes.values()) / len(outcomes)
        comparisons.append(MetricComparison(key, label, probability, len(comparable)))

    values = np.asarray([item.probability for item in comparisons], dtype=np.float64)
    probability = float(np.mean(values))
    standard_deviation = float(np.std(values))
    return WinPrediction(
        probability=probability,
        low=max(0.0, probability - standard_deviation),
        high=min(1.0, probability + standard_deviation),
        completed_seasons=len(outcomes),
        standard_deviation=standard_deviation,
        comparisons=tuple(comparisons),
    )
