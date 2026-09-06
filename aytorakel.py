"""Command line entry point for the AYTOrakel calculator."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from ayto_solver import (
    DataValidationError,
    SCHEMA_VERSION,
    SOLVER_VERSION,
    SeasonSolver,
    SolverState,
    deserialize_event_records,
    deserialize_night_records,
    event_prefix_hash,
    flatten_events,
    serialize_event_records,
    serialize_night_records,
    validate_season_data,
)
from ayto_performance import (
    HISTORY_FILE,
    HistoryError,
    build_performance_records,
    predict_win_chance,
    read_history,
    validate_history,
    write_history,
)
from insta_renderer import (
    POST_SIZE,
    render_combinations_post,
    render_cover,
    render_face_layer_guide,
    render_light_post,
    render_probability_post,
    render_performance_post,
    render_summary,
)


DATA_FILE = Path("ayto_data.json")


def read_all_seasons(path: Path = DATA_FILE) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise DataValidationError(f"Season data file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _artwork_states(data: dict[str, Any], through_week: int | None) -> list[tuple[str, list[str], list[str]]]:
    row_group = data["matrix"]["rows"]
    bit_group = data["matrix"]["bits"]
    groups = {key: list(value) for key, value in data["groups"].items()}
    states = [(data["style"]["face_layer"], list(groups[row_group]), list(groups[bit_group]))]
    for week in data["weeks"]:
        if through_week is not None and int(week["number"]) > through_week:
            break
        for event in week["events"]:
            if event["type"] != "cast_change":
                continue
            for addition in event.get("add", []):
                groups[addition["group"]].append(addition["person"])
                groups[addition["group"]].sort(key=str.casefold)
            if event.get("face_layer"):
                states.append((event["face_layer"], list(groups[row_group]), list(groups[bit_group])))
    return states


def validate_artwork(season: str, data: dict[str, Any], through_week: int | None = None) -> list[Path]:
    missing_guides: list[Path] = []
    background = Path(data["style"]["background"])
    if not background.is_file():
        raise DataValidationError(f"{season}: background not found: {background}")
    from PIL import Image

    with Image.open(background) as image:
        if image.size != POST_SIZE:
            raise DataValidationError(f"{season}: background must be {POST_SIZE}, got {image.size}")
    for layer_name, rows, bits in _artwork_states(data, through_week):
        layer = Path(layer_name)
        if not layer.is_file():
            guide = render_face_layer_guide(season, data["style"], layer, rows, bits)
            missing_guides.append(guide)
            continue
        with Image.open(layer) as image:
            if image.size != POST_SIZE:
                raise DataValidationError(f"{season}: face layer must be {POST_SIZE}, got {image.size}")
    return missing_guides


def _checkpoint_files(season: str) -> list[Path]:
    return sorted(Path(season).glob(".ayto_checkpoint_week_*.npz"))


def save_checkpoint(solver: SeasonSolver, data: dict[str, Any]) -> Path:
    state = solver.state
    if state.solutions is None:
        raise RuntimeError("Cannot checkpoint before solutions have been generated")
    season_dir = Path(state.season)
    season_dir.mkdir(parents=True, exist_ok=True)
    destination = season_dir / f".ayto_checkpoint_week_{state.completed_week:02d}.npz"
    temporary = season_dir / f".{destination.name}.tmp"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "solver_version": SOLVER_VERSION,
        "season": state.season,
        "row_group": state.row_group,
        "bit_group": state.bit_group,
        "groups": state.groups,
        "active": {key: sorted(values, key=state.groups[key].index) for key, values in state.active.items()},
        "multi_matches": state.multi_matches,
        "face_layer": state.face_layer,
        "processed_events": state.processed_events,
        "completed_week": state.completed_week,
        "prefix_hash": event_prefix_hash(data, state.processed_events),
        "night_records": serialize_night_records(state.night_records),
        "event_records": serialize_event_records(state.event_records),
    }
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            solutions=state.solutions,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
    os.replace(temporary, destination)
    for old in _checkpoint_files(state.season):
        if old != destination:
            old.unlink()
    return destination


def load_checkpoint(
    season: str, data: dict[str, Any], through_week: int | None = None
) -> SeasonSolver | None:
    files = _checkpoint_files(season)
    if not files:
        return None
    checkpoint = files[-1]
    try:
        with np.load(checkpoint, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"].item()))
            solutions = saved["solutions"].copy()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Ignoring unreadable checkpoint {checkpoint}: {error}")
        return None
    processed = int(metadata.get("processed_events", -1))
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("solver_version") != SOLVER_VERSION
        or metadata.get("season") != season
        or processed < 0
        or processed > len(flatten_events(data))
        or (
            through_week is not None
            and int(metadata.get("completed_week", 0)) > through_week
        )
        or metadata.get("prefix_hash") != event_prefix_hash(data, processed)
    ):
        print(f"Ignoring stale checkpoint {checkpoint}")
        return None
    solver = SeasonSolver(season, data)
    solver.state = SolverState(
        season=season,
        row_group=metadata["row_group"],
        bit_group=metadata["bit_group"],
        groups={key: list(value) for key, value in metadata["groups"].items()},
        active={key: set(value) for key, value in metadata["active"].items()},
        multi_matches=[dict(rule) for rule in metadata["multi_matches"]],
        face_layer=metadata["face_layer"],
        solutions=solutions,
        processed_events=processed,
        completed_week=int(metadata["completed_week"]),
        night_records=deserialize_night_records(metadata.get("night_records", [])),
        event_records=deserialize_event_records(metadata.get("event_records", [])),
    )
    print(
        f"Resuming {season} after week {solver.state.completed_week} with "
        f"{len(solutions):,} canonical combinations"
    )
    return solver


def _render_new_snapshots(
    season: str,
    data: dict[str, Any],
    solver: SeasonSolver,
    first_snapshot: int,
    output_dir: Path,
) -> list[Path]:
    rendered: list[Path] = []
    for snapshot in solver.snapshots[first_snapshot:]:
        rendered.append(render_probability_post(season, data, snapshot, output_dir))
        if snapshot.light_distribution is not None:
            rendered.append(render_light_post(season, data, snapshot, output_dir))
        combinations = render_combinations_post(season, data, snapshot, output_dir)
        if combinations is not None:
            rendered.append(combinations)
        print(
            f"{season} W{snapshot.week} E{snapshot.event_number} "
            f"{snapshot.title}: {snapshot.solution_count:,} combinations"
        )
    return rendered


def run_season(
    season: str,
    data: dict[str, Any],
    *,
    from_scratch: bool = False,
    through_week: int | None = None,
    prevalidated: bool = False,
    historical_records: list | None = None,
) -> None:
    if not prevalidated:
        validate_season_data(season, data)
        guides = validate_artwork(season, data, through_week)
        if guides:
            rendered = "\n".join(f"  - {path}" for path in guides)
            raise DataValidationError(
                f"{season}: missing face layer(s). Layout guide(s) were created:\n{rendered}"
            )
    output_dir = Path(season)
    output_dir.mkdir(parents=True, exist_ok=True)
    solver = None if from_scratch else load_checkpoint(season, data, through_week)
    if solver is None:
        solver = SeasonSolver(season, data)
    generated: set[Path] = set()

    processed = solver.state.processed_events
    cumulative = 0
    for week in data["weeks"]:
        week_number = int(week["number"])
        week_length = len(week["events"])
        week_end = cumulative + week_length
        if through_week is not None and week_number > through_week:
            break
        if processed >= week_end:
            cumulative = week_end
            continue
        offset = max(0, processed - cumulative)
        remaining_week = {"number": week_number, "events": week["events"][offset:]}
        if offset == 0:
            generated.add(render_cover(season, week_number, data["style"], output_dir))
        snapshot_start = len(solver.snapshots)
        solver.process_week(remaining_week, processed, event_number_offset=offset)
        processed = solver.state.processed_events
        generated.update(
            _render_new_snapshots(season, data, solver, snapshot_start, output_dir)
        )
        if data["status"] == "ongoing":
            if historical_records is None:
                raise HistoryError(
                    "Historisches Dataset fehlt. Bitte zuerst "
                    "`python aytorakel.py --build-history --from-scratch` ausführen."
                )
            live_records = build_performance_records(
                season, data, solver.state.event_records
            )
            current = live_records[-1]
            model_history = [
                row for row in historical_records if row.status == "completed"
            ]
            prediction = predict_win_chance(model_history, current)
            generated.add(
                render_performance_post(
                    season,
                    data,
                    current,
                    [*model_history, *live_records],
                    prediction,
                    output_dir,
                )
            )
        checkpoint = save_checkpoint(solver, data)
        print(f"Saved checkpoint {checkpoint}")
        cumulative = week_end

    last_week = solver.state.completed_week or (through_week or 1)
    summary = render_summary(season, data, solver.state.night_records, output_dir, last_week)
    if summary is not None:
        generated.add(summary)

    if from_scratch and through_week is None:
        for old in output_dir.iterdir():
            if (
                old.is_file()
                and old.suffix.lower() in {".png", ".mp4"}
                and old not in generated
            ):
                old.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate exact AYTO probabilities and Instagram posts")
    parser.add_argument("season", nargs="?", help="Season key, for example s6vip")
    parser.add_argument("--all", action="store_true", help="Process every season")
    parser.add_argument("--from-scratch", action="store_true", help="Ignore compatible checkpoints")
    parser.add_argument("--through-week", type=int, help="Stop after this week")
    parser.add_argument("--validate", action="store_true", help="Validate data and artwork without calculating")
    parser.add_argument(
        "--build-history",
        action="store_true",
        help="Recalculate all seasons without rendering and write historical_metrics.csv",
    )
    return parser


def build_history_dataset(
    seasons: dict[str, dict[str, Any]], *, from_scratch: bool = False
) -> list:
    records = []
    for season, data in seasons.items():
        validate_season_data(season, data)
        solver = None if from_scratch else load_checkpoint(season, data)
        if solver is None:
            solver = SeasonSolver(season, data)
        processed = solver.state.processed_events
        cumulative = 0
        for week in data["weeks"]:
            week_length = len(week["events"])
            week_end = cumulative + week_length
            if processed < week_end:
                offset = max(0, processed - cumulative)
                solver.process_week(
                    {"number": week["number"], "events": week["events"][offset:]},
                    processed,
                    event_number_offset=offset,
                )
                processed = solver.state.processed_events
            cumulative = week_end
        season_records = build_performance_records(
            season, data, solver.state.event_records
        )
        records.extend(season_records)
        print(
            f"Historie {season}: {len(season_records)} Wochen, "
            f"{season_records[-1].solutions_remaining:,} Kombinationen"
        )
    write_history(records, HISTORY_FILE)
    print(f"Historisches Dataset geschrieben: {HISTORY_FILE}")
    return records


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seasons = read_all_seasons()
        if args.build_history:
            if args.season or args.all or args.through_week or args.validate:
                raise DataValidationError(
                    "--build-history wird ohne Season, --all, --through-week oder --validate verwendet"
                )
            build_history_dataset(seasons, from_scratch=args.from_scratch)
            return 0
        if args.all:
            selected = list(seasons)
        elif args.season:
            if args.season not in seasons:
                raise DataValidationError(
                    f"Unknown season {args.season!r}; choose one of {', '.join(seasons)}"
                )
            selected = [args.season]
        else:
            raise DataValidationError("Provide a season or use --all")

        historical_records = None
        if not args.validate and any(
            seasons[season]["status"] == "ongoing" for season in selected
        ):
            try:
                historical_records = read_history(HISTORY_FILE)
                validate_history(historical_records, seasons)
            except HistoryError as error:
                raise HistoryError(
                    f"{error}. Bitte zuerst `python aytorakel.py "
                    "--build-history --from-scratch` ausführen."
                ) from error

        missing: list[str] = []
        for season in selected:
            validate_season_data(season, seasons[season])
            guides = validate_artwork(season, seasons[season], args.through_week)
            missing.extend(f"{season}: {guide}" for guide in guides)
        if missing:
            raise DataValidationError(
                "Missing face layers; generated guides:\n  - " + "\n  - ".join(missing)
            )
        if args.validate:
            for season in selected:
                print(f"Validated {season}")
            return 0

        for season in selected:
            run_season(
                season,
                seasons[season],
                from_scratch=args.from_scratch,
                through_week=args.through_week,
                prevalidated=True,
                historical_records=historical_records,
            )
        return 0
    except (DataValidationError, HistoryError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
