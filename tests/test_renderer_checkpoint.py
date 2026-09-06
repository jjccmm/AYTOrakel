import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from ayto_solver import NightRecord, SeasonSolver, Snapshot
from aytorakel import load_checkpoint, run_season, save_checkpoint
from insta_renderer import (
    POST_SIZE,
    _annotation_color,
    _probability_label,
    render_combinations_post,
    render_light_post,
    render_probability_post,
    render_summary,
)
from test_solver import toy_data


class RendererTests(unittest.TestCase):
    def test_heatmap_labels_are_precise_and_contrasting(self):
        self.assertEqual("", _probability_label(0))
        self.assertEqual("0.004", _probability_label(0.0044))
        self.assertEqual("<0.001", _probability_label(0.0001))
        self.assertEqual("100", _probability_label(100))
        self.assertEqual("white", _annotation_color(1))
        self.assertEqual("#262626", _annotation_color(100))

    def test_final_posts_have_instagram_size_and_no_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            background = root / "background.png"
            faces = root / "faces.png"
            Image.new("RGB", POST_SIZE, "#111111").save(background)
            Image.new("RGBA", POST_SIZE, (0, 0, 0, 0)).save(faces)
            data = toy_data(["A", "B"], ["x", "y"])
            data["style"] = {"background": str(background), "face_layer": str(faces)}
            snapshot = Snapshot(
                week=1,
                event_number=1,
                event_index=-1,
                event_type="matching_night",
                title="Matching Night",
                solution_count=2,
                probabilities=np.full((2, 2), 50.0),
                row_names=("A", "B"),
                bit_names=("x", "y"),
                active={"a": ("A", "B"), "b": ("x", "y")},
                face_layer=str(faces),
                light_distribution=np.asarray([0.5, 0.0, 0.5]),
                actual_lights=2,
                combination_sample=np.asarray([[1, 2], [2, 1]], dtype=np.uint16),
            )
            paths = [
                render_probability_post("toy", data, snapshot, root),
                render_light_post("toy", data, snapshot, root),
                render_combinations_post("toy", data, snapshot, root),
            ]
            self.assertTrue(all(path is not None for path in paths))
            for path in paths:
                with Image.open(path) as image:
                    self.assertEqual(POST_SIZE, image.size)
            self.assertEqual({path.name for path in paths}, {path.name for path in root.glob("toy*.png")})

    def test_more_than_24_combinations_skips_remaining_post(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = Snapshot(
                1, 1, -1, "box", "Match Box", 25,
                np.zeros((1, 1)), ("A",), ("x",), {"a": ("A",), "b": ("x",)},
                "unused.png", combination_sample=None,
            )
            self.assertIsNone(render_combinations_post("toy", {}, snapshot, root))

    def test_summary_is_only_created_after_week_ten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            background = root / "background.png"
            Image.new("RGB", POST_SIZE, "#111111").save(background)
            data = toy_data(
                ["A", "B"],
                ["x", "y"],
                weeks=[
                    {
                        "number": 1,
                        "events": [
                            {"type": "box", "pair": ["A", "x"], "result": "yes"}
                        ],
                    }
                ],
            )
            data["style"] = {"background": str(background), "face_layer": "unused.png"}
            records = [
                NightRecord(
                    week=1,
                    event_number=2,
                    distribution=np.asarray([0.25, 0.50, 0.25]),
                    actual_lights=1,
                )
            ]

            self.assertIsNone(render_summary("toy", data, records, root, 9))
            path = render_summary("toy", data, records, root, 10)
            self.assertIsNotNone(path)
            self.assertEqual("toy_11_0_insta_summary.png", path.name)
            with Image.open(path) as image:
                self.assertEqual(POST_SIZE, image.size)


class CheckpointTests(unittest.TestCase):
    def test_resume_matches_clean_run_and_rejects_edited_prefix(self):
        weeks = [
            {"number": 1, "events": [{"type": "box", "pair": ["A", "x"], "result": "no"}]},
            {"number": 2, "events": [{"type": "box", "pair": ["B", "x"], "result": "yes"}]},
        ]
        data = toy_data(["A", "B"], ["x", "y"], weeks=weeks)
        with tempfile.TemporaryDirectory() as directory:
            season = str(Path(directory) / "toy")
            partial = SeasonSolver(season, data)
            partial.process_week(weeks[0], 0)
            save_checkpoint(partial, data)
            resumed = load_checkpoint(season, data)
            self.assertIsNotNone(resumed)
            resumed.process_week(weeks[1], 1)

            clean = SeasonSolver("clean", data)
            clean.process_week(weeks[0], 0)
            clean.process_week(weeks[1], 1)
            np.testing.assert_array_equal(resumed.state.solutions, clean.state.solutions)
            self.assertEqual(resumed.state.event_records, clean.state.event_records)

            edited = toy_data(["A", "B"], ["x", "y"], weeks=[
                {"number": 1, "events": [{"type": "box", "pair": ["A", "x"], "result": "yes"}]},
                weeks[1],
            ])
            self.assertIsNone(load_checkpoint(season, edited))

    def test_same_week_appended_event_resumes_at_its_real_event_number(self):
        first_event = {"type": "box", "pair": ["A", "x"], "result": "no"}
        data = toy_data(
            ["A", "B"], ["x", "y"],
            weeks=[{"number": 1, "events": [first_event]}],
        )
        with tempfile.TemporaryDirectory() as directory:
            season = str(Path(directory) / "toy")
            partial = SeasonSolver(season, data)
            partial.process_week(data["weeks"][0], 0)
            save_checkpoint(partial, data)

            appended = toy_data(
                ["A", "B"], ["x", "y"],
                weeks=[{"number": 1, "events": [
                    first_event,
                    {"type": "box", "pair": ["B", "x"], "result": "yes"},
                ]}],
            )
            resumed = load_checkpoint(season, appended)
            self.assertIsNotNone(resumed)
            resumed.process_week(
                {"number": 1, "events": appended["weeks"][0]["events"][1:]},
                1,
                event_number_offset=1,
            )
            self.assertEqual(2, resumed.snapshots[-1].event_number)
            self.assertEqual(2, resumed.state.processed_events)

    def test_missing_artwork_stops_before_solver_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            background = root / "background.png"
            Image.new("RGB", POST_SIZE).save(background)
            data = toy_data(["A", "B"], ["x", "y"], weeks=[])
            data["style"] = {
                "background": str(background),
                "face_layer": str(root / "missing.png"),
            }
            guide = root / "guide.png"
            with patch("aytorakel.render_face_layer_guide", return_value=guide), patch(
                "aytorakel.SeasonSolver"
            ) as solver:
                with self.assertRaisesRegex(ValueError, "missing face layer"):
                    run_season("toy", data)
                solver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
