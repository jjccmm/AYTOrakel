import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ayto_performance import (
    METRIC_VERSION,
    HistoryError,
    MetricComparison,
    PerformanceRecord,
    WinPrediction,
    build_performance_records,
    predict_win_chance,
    read_history,
    validate_history,
    write_history,
)
from ayto_solver import (
    EventCountRecord,
    SeasonSolver,
    apply_evidence_event,
    flatten_events,
)
from insta_renderer import render_performance_post


def season_data(weeks):
    return {
        "schema_version": 2,
        "status": "ongoing",
        "outcome": None,
        "competitive_end_week": None,
        "target_lights": 2,
        "initial_active": "all",
        "groups": {"a": ["A", "B"], "b": ["x", "y"]},
        "matrix": {"rows": "a", "bits": "b"},
        "initial_match_model": {"multi_matches": []},
        "style": {"background": "unused.png", "face_layer": "unused.png"},
        "weeks": weeks,
    }


class MetricTests(unittest.TestCase):
    def test_information_efficiency_and_sold_events(self):
        data = season_data([{"number": 1, "events": [
            {"type": "box", "pair": ["A", "x"], "result": "no"},
            {"type": "matching_night", "result": "sold", "pairs": [["A", "y"], ["B", "x"]]},
        ]}])
        counts = [
            EventCountRecord(1, 0, -1, "entry", 2, 2),
            EventCountRecord(1, 1, 0, "box", 2, 1),
            EventCountRecord(1, 2, 1, "matching_night", 1, 1),
        ]
        record = build_performance_records("toy", data, counts)[0]
        self.assertEqual(record.solutions_remaining, 1)
        self.assertEqual(record.information_bits, 1)
        self.assertEqual(record.information_progress, 1)
        self.assertEqual(record.decision_count, 2)
        self.assertAlmostEqual(record.mean_elimination, 1 - 2 ** -0.5)
        self.assertEqual(record.observed_nights, 0)
        self.assertEqual(record.light_form, 0)
        self.assertEqual(record.box_information_gain, 0.5)
        self.assertEqual(record.night_information_gain, 0)

    def test_progress_can_drop_only_at_structural_growth(self):
        data = season_data([
            {"number": 1, "events": [{"type": "box", "pair": ["A", "x"], "result": "no"}]},
            {"number": 2, "events": [{
                "type": "cast_change", "add": [{"group": "b", "person": "z"}],
                "match_update": {"mode": "rebuild", "multi_matches": [{"group": "bits", "size": 2}]},
            }]},
        ])
        counts = [
            EventCountRecord(1, 0, -1, "entry", 24, 24),
            EventCountRecord(1, 1, 0, "box", 24, 4),
            EventCountRecord(2, 1, 1, "cast_change", 12, 12),
        ]
        records = build_performance_records("toy", data, counts)
        self.assertGreater(records[0].information_progress, records[1].information_progress)
        self.assertAlmostEqual(records[0].information_bits, records[1].information_bits)

    def test_lights_ewma_and_disjoint_secure_matches(self):
        data = season_data([{"number": 1, "events": [
            {"type": "reveal", "complete_groups": [
                {"partner": "A", "members": ["x", "y"], "complete": True}
            ]},
            {"type": "box", "pair": ["B", "x"], "result": "yes", "complete_pair": False},
            {"type": "matching_night", "lights": 1, "pairs": [["A", "y"], ["B", "x"]]},
            {"type": "matching_night", "lights": 2, "pairs": [["A", "y"], ["B", "x"]]},
        ]}])
        counts = [EventCountRecord(1, 0, -1, "entry", 8, 8)] + [
            EventCountRecord(1, i + 1, i, event["type"], 8, 8)
            for i, event in enumerate(data["weeks"][0]["events"])
        ]
        record = build_performance_records("toy", data, counts)[0]
        self.assertEqual(record.secure_matches, 2)
        self.assertEqual(record.light_form, 0.75)
        self.assertEqual(record.confirmed_boxes, 1)
        self.assertTrue(record.double_match_found)

    def test_sold_night_does_not_filter_or_create_distribution(self):
        data = season_data([])
        solver = SeasonSolver("toy", data)
        solver.state.solutions = np.asarray([[1, 2], [2, 1]], dtype=np.uint16)
        event = {"type": "matching_night", "result": "sold", "pairs": [["A", "x"], ["B", "y"]]}
        survivors, distribution = apply_evidence_event(solver.state, solver.state.solutions, event)
        np.testing.assert_array_equal(survivors, solver.state.solutions)
        self.assertIsNone(distribution)


def model_record(
    season, week, outcome, progress, lights, secure, terminal=False, status="completed",
    *, week_lights=2, cumulative_lights=6, confirmed_boxes=1,
):
    return PerformanceRecord(
        season=season, week=week, status=status, outcome=outcome,
        competitive_end_week=3 if status == "completed" else None, target_lights=10,
        source_hash="hash", event_count=week, solutions_remaining=100,
        information_bits=progress, information_progress=progress,
        decision_count=week * 2, mean_elimination=0.4, observed_nights=3,
        latest_lights=week_lights, week_lights=week_lights,
        cumulative_lights=cumulative_lights, light_form=lights,
        secure_matches=secure, confirmed_boxes=confirmed_boxes,
        double_match_found=False, box_information_gain=0.2,
        night_information_gain=0.3, terminal=terminal, metric_version=METRIC_VERSION,
    )


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            model_record("s1", 3, "won", .8, .3, 1, week_lights=1, cumulative_lights=9, confirmed_boxes=4),
            model_record("s2", 3, "lost", .2, .3, 0, week_lights=2, cumulative_lights=4, confirmed_boxes=3),
            model_record("s3", 3, "won", .7, .3, 1, week_lights=3, cumulative_lights=5, confirmed_boxes=1),
            model_record("s4", 3, "lost", .4, .3, 1, week_lights=4, cumulative_lights=6, confirmed_boxes=2),
        ]
        self.target = model_record(
            "ongoing", 3, "", .4, .3, 1, status="ongoing",
            week_lights=2, cumulative_lights=5, confirmed_boxes=1,
        )

    def test_same_or_worse_comparisons_are_averaged(self):
        prediction = predict_win_chance(self.records, self.target)
        self.assertEqual([item.probability for item in prediction.comparisons], [.5, .5, 1, 0])
        self.assertEqual([item.comparable_seasons for item in prediction.comparisons], [2, 2, 1, 2])
        self.assertEqual(prediction.probability, .5)
        self.assertAlmostEqual(prediction.standard_deviation, math.sqrt(.125))
        self.assertTrue(0 <= prediction.low <= prediction.high <= 1)


class CsvAndRendererTests(unittest.TestCase):
    def test_ongoing_season_may_be_newer_than_history(self):
        data = season_data([{"number": 1, "events": []}])
        from ayto_performance import season_source_hash
        record = model_record("toy", 1, "", 0.2, 0.3, 1, status="ongoing")
        record = PerformanceRecord(**{**record.__dict__, "source_hash": season_source_hash(data)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            write_history([record], path)
            loaded = read_history(path)
            self.assertEqual(loaded, [record])
            validate_history(loaded, {"toy": data})
            data["target_lights"] = 3
            validate_history(loaded, {"toy": data})

    def test_completed_season_with_stale_hash_requires_history_rebuild(self):
        data = season_data([{"number": 1, "events": []}])
        data["status"] = "completed"
        data["outcome"] = "won"
        from ayto_performance import season_source_hash
        record = model_record("toy", 1, "won", 0.2, 0.3, 1, status="completed")
        record = PerformanceRecord(**{**record.__dict__, "source_hash": season_source_hash(data)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            write_history([record], path)
            loaded = read_history(path)
            validate_history(loaded, {"toy": data})
            data["target_lights"] = 3
            with self.assertRaises(HistoryError):
                validate_history(loaded, {"toy": data})

    def test_performance_post_is_instagram_sized_and_has_no_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            background = root / "background.png"
            Image.new("RGB", (1080, 1350), "#24132b").save(background)
            data = season_data([{"number": 1, "events": [{"type": "box"}, {"type": "matching_night"}]}])
            data["style"]["background"] = str(background)
            current = model_record("current", 1, "", 0.4, 0.5, 2, status="ongoing")
            history = [current]
            for i in range(4):
                history.append(model_record(f"old{i}", 1, "won" if i < 2 else "lost", 0.1 + i * 0.1, 0.3, 1))
            comparisons = tuple(
                MetricComparison(str(i), label, value, i + 2)
                for i, (label, value) in enumerate([
                    ("Lichter dieser Nacht", .5), ("Lichter insgesamt", .6),
                    ("Bestätigte Match Boxes", .7), ("Rätsel gelöst", .4),
                ])
            )
            prediction = WinPrediction(0.55, 0.40, 0.70, 12, 0.15, comparisons)
            path = render_performance_post("current", data, current, history, prediction, root)
            with Image.open(path) as image:
                self.assertEqual(image.size, (1080, 1350))
            self.assertEqual([item for item in root.iterdir() if item.name.startswith(".")], [])
            self.assertEqual(path.name, "current_1_3_insta_Performance.png")


class S7RegressionTests(unittest.TestCase):
    def test_reunion_solution_is_consistent_and_contains_noel_double(self):
        data = json.loads(Path("ayto_data.json").read_text(encoding="utf-8"))["s7"]
        solver = SeasonSolver("s7", data)
        expected = np.zeros((1, len(solver.state.row_names)), dtype=np.uint16)
        solution = data["weeks"][-1]["events"][0]
        for pair in solution["pairs"]:
            row, bit = pair if pair[0] in solver.state.row_names else pair[::-1]
            expected[0, solver.state.row_names.index(row)] |= 1 << solver.state.bit_names.index(bit)
        current = expected
        for _, _, event in flatten_events(data):
            if event["type"] == "cast_change":
                for person in event.get("remove", []):
                    for active in solver.state.active.values():
                        active.discard(person)
            current, _ = apply_evidence_event(solver.state, current, event)
            self.assertEqual(len(current), 1)
        noel = int(current[0, solver.state.row_names.index("Noel")])
        self.assertTrue(noel & (1 << solver.state.bit_names.index("Alicia")))
        self.assertTrue(noel & (1 << solver.state.bit_names.index("Tonia")))


if __name__ == "__main__":
    unittest.main()
