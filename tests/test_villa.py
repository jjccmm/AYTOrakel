import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import insta_renderer
from PIL import Image, ImageColor, ImageDraw

from ayto_solver import DataValidationError, SeasonSolver, flatten_events
from ayto_villa import build_villa_snapshot, iter_villa_snapshots
from aytorakel import run_season, validate_artwork
from insta_renderer import (
    POST_SIZE,
    VILLA_CELL_COLOR,
    VILLA_PAPER_COLOR,
    _villa_label_font,
    heatmap_bounds,
    render_villa_post,
    villa_table_bounds,
)
from test_solver import toy_data


def night(pairs, lights=None, **extra):
    event = {"type": "matching_night", "pairs": pairs, **extra}
    if lights is not None:
        event["lights"] = lights
    return event


def box(a, b, result, **extra):
    return {"type": "box", "pair": [a, b], "result": result, **extra}


def week(number, *events):
    return {"number": number, "events": list(events)}


def ten_pairs():
    return [[f"A{i}", f"B{i}"] for i in range(10)]


def ten_data(*weeks):
    pairs = ten_pairs()
    return toy_data(
        [a for a, _ in pairs], [b for _, b in pairs], weeks=list(weeks)
    )


class VillaKnowledgeTests(unittest.TestCase):
    def test_no_match_reduces_denominator_and_keeps_direct_source(self):
        data = ten_data(
            week(1, night(ten_pairs(), 4)),
            week(2, box("B0", "A0", "no"), night(ten_pairs(), 4)),
        )
        first = build_villa_snapshot(data, 1)
        second = build_villa_snapshot(data, 2)
        self.assertEqual("4/10", first.nights[0].label)
        self.assertNotIn(("A0", "B0"), first.pair_states)
        self.assertEqual("4/9", second.nights[0].label)
        self.assertEqual(("no", "box"), (
            second.pair_states[("A0", "B0")].result,
            second.pair_states[("A0", "B0")].source,
        ))

    def test_perfect_match_reduces_numerator_and_denominator(self):
        data = ten_data(
            week(1, night(ten_pairs(), 4)),
            week(2, box("A0", "B0", "yes")),
        )
        snapshot = build_villa_snapshot(data, 2)
        self.assertEqual("3/9", snapshot.nights[0].label)
        self.assertEqual("yes", snapshot.pair_states[("A0", "B0")].result)
        for pair in (("A0", "B1"), ("A1", "B0")):
            self.assertEqual("partners", snapshot.pair_states[pair].source)

    def test_confirmed_pair_outside_night_removes_two_old_pairs(self):
        data = ten_data(
            week(1, night(ten_pairs(), 4)),
            week(2, box("A0", "B1", "yes")),
        )
        snapshot = build_villa_snapshot(data, 2)
        self.assertEqual("4/8", snapshot.nights[0].label)

    def test_blackout_excludes_all_seated_pairs(self):
        data = ten_data(week(1, box("A0", "B0", "no"), night(ten_pairs(), 0)))
        snapshot = build_villa_snapshot(data, 1)
        self.assertEqual("box", snapshot.pair_states[("A0", "B0")].source)
        self.assertEqual("matching_night", snapshot.pair_states[("A1", "B1")].source)
        self.assertTrue(all(s.result == "no" for s in snapshot.pair_states.values()))
        self.assertIsNone(snapshot.nights[0].label)
        self.assertEqual(0, snapshot.nights[0].unresolved_lights)

    def test_full_night_confirms_pairs_and_excludes_partners(self):
        snapshot = build_villa_snapshot(ten_data(week(1, night(ten_pairs(), 10))), 1)
        self.assertEqual(100, len(snapshot.pair_states))
        self.assertEqual(10, sum(s.result == "yes" for s in snapshot.pair_states.values()))
        self.assertIsNone(snapshot.nights[0].label)

    def test_resolution_propagates_between_nights_to_fixed_point(self):
        data = toy_data(
            ["A", "B", "C"], ["x", "y", "z"],
            weeks=[week(
                1,
                night([["A", "x"], ["B", "y"], ["C", "z"]], 1),
                night([["A", "y"], ["B", "z"], ["C", "x"]], 1),
                box("B", "y", "no"), box("C", "z", "no"),
            )],
        )
        snapshot = build_villa_snapshot(data, 1)
        for pair in (("A", "x"), ("B", "z")):
            self.assertEqual("yes", snapshot.pair_states[pair].result)
            self.assertEqual("matching_night", snapshot.pair_states[pair].source)
        self.assertTrue(all(n.label is None for n in snapshot.nights))
        # No extra singleton or solution-enumeration inference is performed.
        self.assertNotIn(("C", "y"), snapshot.pair_states)

    def test_automatic_lights_and_explicit_confirmations_count_once(self):
        data = ten_data(week(
            1,
            box("A0", "B0", "yes"),
            night(ten_pairs(), 5, automatic_lights=1),
        ))
        self.assertEqual("3/9", build_villa_snapshot(data, 1).nights[0].label)

    def test_sold_events_stay_unknown_and_keep_night_number(self):
        data = ten_data(
            week(1, box("A0", "B0", "sold"), night(ten_pairs(), result="sold")),
            week(2, night(ten_pairs(), 4)),
        )
        snapshot = build_villa_snapshot(data, 2)
        self.assertEqual(["?", "4/10"], [n.label for n in snapshot.nights])
        self.assertEqual([1, 2], [n.number for n in snapshot.nights])
        self.assertNotIn(("A0", "B0"), snapshot.pair_states)

    def test_repeated_pair_keeps_separate_unreduced_fractions(self):
        pairs = ten_pairs()
        data = ten_data(
            week(1, night(pairs, 4)),
            week(2, night(pairs[:6], 3)),
        )
        snapshot = build_villa_snapshot(data, 2)
        self.assertEqual(["4/10", "3/6"], [n.label for n in snapshot.nights])
        self.assertIn(("A0", "B0"), snapshot.nights[0].open_pairs)
        self.assertIn(("A0", "B0"), snapshot.nights[1].open_pairs)

    def test_partial_double_match_does_not_close_shared_partner(self):
        data = toy_data(
            ["A", "B"], ["x", "y", "z"],
            multi=[{"group": "bits", "size": 2, "known_members": []}],
            weeks=[week(1, box("A", "x", "yes", complete_pair=False))],
        )
        snapshot = build_villa_snapshot(data, 1)
        self.assertNotIn(("A", "y"), snapshot.pair_states)
        self.assertEqual("no", snapshot.pair_states[("B", "x")].result)
        data["weeks"].append(week(2, box("A", "y", "yes", complete_pair=False)))
        snapshot = build_villa_snapshot(data, 2)
        self.assertEqual("no", snapshot.pair_states[("A", "z")].result)

    def test_double_match_on_transposed_side(self):
        data = toy_data(
            ["A", "B", "C"], ["x", "y"],
            multi=[{"group": "rows", "size": 2}],
            weeks=[week(1, box("A", "x", "yes", complete_pair=False))],
        )
        snapshot = build_villa_snapshot(data, 1)
        self.assertNotIn(("B", "x"), snapshot.pair_states)
        self.assertEqual("no", snapshot.pair_states[("A", "y")].result)
        data["weeks"].append(week(2, box("B", "x", "yes", complete_pair=False)))
        self.assertEqual(
            "no", build_villa_snapshot(data, 2).pair_states[("C", "x")].result
        )

    def test_complete_double_group_is_revealed_and_closed(self):
        data = toy_data(
            ["A", "B"], ["x", "y", "z"],
            multi=[{"group": "bits", "size": 2, "known_members": []}],
            weeks=[week(1, box(
                "A", "x", "yes",
                revealed_group={"partner": "A", "members": ["x", "y"], "complete": True},
            ))],
        )
        snapshot = build_villa_snapshot(data, 1)
        self.assertEqual("yes", snapshot.pair_states[("A", "y")].result)
        for pair in (("A", "z"), ("B", "x"), ("B", "y")):
            self.assertEqual("no", snapshot.pair_states[pair].result)

    def test_complete_normal_box_still_closes_in_double_season(self):
        data = toy_data(
            ["A", "B"], ["x", "y", "z"],
            multi=[{"group": "bits", "size": 2, "known_members": []}],
            weeks=[week(1, box("A", "x", "yes"))],
        )
        self.assertEqual(
            "no", build_villa_snapshot(data, 1).pair_states[("A", "y")].result
        )

    def test_triple_match_preserves_capacity_until_third_confirmation(self):
        data = toy_data(
            ["A", "B"], ["w", "x", "y", "z"],
            multi=[{"group": "bits", "size": 3, "known_members": []}],
            weeks=[week(
                1, box("A", "x", "yes", complete_pair=False),
                box("A", "y", "yes", complete_pair=False),
            )],
        )
        self.assertNotIn(("A", "z"), build_villa_snapshot(data, 1).pair_states)
        data["weeks"].append(week(2, box("A", "z", "yes", complete_pair=False)))
        self.assertEqual(
            "no", build_villa_snapshot(data, 2).pair_states[("A", "w")].result
        )

    def test_newcomer_recomputes_inferred_exclusivity_and_layout(self):
        data = toy_data(
            ["A", "B", "C"], ["x", "y", "z"],
            weeks=[
                week(1, night([["A", "x"], ["B", "y"], ["C", "z"]], 3)),
                week(2, {
                    "type": "cast_change", "add": [{"group": "b", "person": "a"}],
                    "face_layer": "new.png",
                    "match_update": {
                        "mode": "extend",
                        "new_multi_match": {
                            "group": "bits", "size": 2, "known_members": ["a"],
                        },
                        "eligible_partners": ["A", "B"],
                    },
                }, night([["A", "a"], ["B", "y"], ["C", "z"]], 2)),
            ],
        )
        first = build_villa_snapshot(data, 1)
        second = build_villa_snapshot(data, 2)
        self.assertEqual(("x", "y", "z"), first.bit_names)
        self.assertEqual(("a", "x", "y", "z"), second.bit_names)
        self.assertEqual("new.png", second.face_layer)
        self.assertEqual("yes", second.pair_states[("A", "x")].result)
        self.assertEqual("matching_night", second.pair_states[("A", "a")].source)
        self.assertEqual("no", second.pair_states[("A", "a")].result)
        # Without the new night, the extra partner must remain open.
        data["weeks"][1]["events"].pop()
        self.assertNotIn(("A", "a"), build_villa_snapshot(data, 2).pair_states)
        self.assertEqual("no", build_villa_snapshot(data, 2).pair_states[("C", "a")].result)

    def test_rebuild_allows_new_double_without_old_inferred_exclusions(self):
        data = toy_data(
            ["A", "B"], ["x", "y"],
            weeks=[
                week(1, night([["A", "x"], ["B", "y"]], 2)),
                week(2, {
                    "type": "cast_change", "add": [{"group": "b", "person": "z"}],
                    "match_update": {
                        "mode": "rebuild",
                        "multi_matches": [{"group": "bits", "size": 2}],
                    },
                }),
            ],
        )
        self.assertNotIn(("A", "z"), build_villa_snapshot(data, 2).pair_states)

    def test_public_reveal_and_solution_close_known_matches(self):
        data = toy_data(
            ["A", "B"], ["x", "y"],
            weeks=[
                week(1, {"type": "reveal", "pairs": [["A", "x"]]}),
                week(2, {"type": "solution", "pairs": [["A", "x"], ["B", "y"]]}),
            ],
        )
        self.assertEqual("reveal", build_villa_snapshot(data, 1).pair_states[("A", "x")].source)
        self.assertEqual(4, len(build_villa_snapshot(data, 2).pair_states))

    def test_week_iteration_and_tenth_night(self):
        data = ten_data(
            week(1, night(ten_pairs(), 4)),
            week(2, box("A0", "B0", "sold")),
            week(3, night(ten_pairs(), 4), night(ten_pairs(), 4)),
        )
        self.assertEqual([1, 3], [s.week for s in iter_villa_snapshots(data)])
        self.assertEqual([1], [s.week for s in iter_villa_snapshots(data, 2)])
        self.assertEqual(3, build_villa_snapshot(data, 3).night_rows)
        data = ten_data(*(week(i, night(ten_pairs(), 4)) for i in range(1, 11)))
        self.assertEqual(3, build_villa_snapshot(data, 9).night_rows)
        self.assertEqual(4, build_villa_snapshot(data, 10).night_rows)

    def test_contradictory_light_counts_are_rejected(self):
        data = ten_data(week(1, box("A0", "B0", "yes"), night(ten_pairs(), 0)))
        with self.assertRaisesRegex(DataValidationError, "MN 1"):
            build_villa_snapshot(data, 1)
        data = ten_data(week(1, box("A0", "B0", "no"), night(ten_pairs(), 10)))
        with self.assertRaisesRegex(DataValidationError, "MN 1"):
            build_villa_snapshot(data, 1)

    def test_no_solution_enumeration_and_no_input_mutation(self):
        data = ten_data(week(1, night(ten_pairs(), 4)))
        original = copy.deepcopy(data)
        with patch("ayto_solver.generate_solution_batches", side_effect=AssertionError):
            build_villa_snapshot(data, 1)
        self.assertEqual(original, data)

    def test_deductions_are_true_in_all_solutions_of_small_fixtures(self):
        fixtures = [
            toy_data(
                ["A", "B", "C"], ["x", "y", "z"],
                weeks=[week(
                    1, night([["A", "x"], ["B", "y"], ["C", "z"]], 1),
                    box("B", "y", "no"), box("C", "z", "no"),
                    night([["A", "y"], ["B", "z"], ["C", "x"]], 1),
                )],
            ),
            toy_data(
                ["A", "B"], ["x", "y", "z"],
                multi=[{"group": "bits", "size": 2}],
                weeks=[week(
                    1, box("A", "x", "yes", complete_pair=False),
                    night([["A", "y"], ["B", "z"]], 2),
                )],
            ),
        ]
        for data in fixtures:
            with self.subTest(data=data):
                solver = SeasonSolver("toy", data)
                solver.process_week(data["weeks"][0], 0)
                snapshot = build_villa_snapshot(data, 1)
                for (a, b), state in snapshot.pair_states.items():
                    row = solver.state.row_names.index(a)
                    bit = solver.state.bit_names.index(b)
                    present = (solver.state.solutions[:, row] & (1 << bit)) != 0
                    self.assertTrue(np.all(present if state.result == "yes" else ~present))

    def test_all_repository_week_prefixes_are_consistent(self):
        data = json.loads(Path("ayto_data.json").read_text(encoding="utf-8"))
        for season, season_data in data.items():
            with self.subTest(season=season):
                for snapshot in iter_villa_snapshots(season_data):
                    for record in snapshot.nights:
                        if record.unresolved_lights is not None:
                            self.assertTrue(0 <= record.unresolved_lights <= len(record.open_pairs))


class VillaTableKnowledgeTests(unittest.TestCase):
    def test_fixed_cast_columns_and_ten_rows_from_first_week(self):
        data = json.loads(Path("ayto_data.json").read_text(encoding="utf-8"))["s6vip"]
        first = build_villa_snapshot(data, 1)
        latest = build_villa_snapshot(data, 8)
        self.assertEqual(11, len(first.table_columns))
        self.assertIn("Laurenz", first.table_columns)
        self.assertEqual(first.table_columns, latest.table_columns)
        self.assertEqual(list(range(1, 11)), [row.number for row in first.table_rows])
        self.assertTrue(first.table_rows[0].played)
        self.assertTrue(all(not row.played and not row.seatings for row in first.table_rows[1:]))
        self.assertEqual(2, len(first.matchboxes))
        self.assertEqual(9, len(latest.matchboxes))
        self.assertNotIn("Laurenz", first.table_rows[0].seatings)

    def test_confirmed_match_prefills_future_and_automatic_seats(self):
        data = ten_data(
            week(1, night(ten_pairs(), 4)),
            week(2, box("A0", "B0", "yes"), night(ten_pairs()[1:], 4, automatic_lights=1)),
        )
        first = build_villa_snapshot(data, 1)
        second = build_villa_snapshot(data, 2)
        self.assertEqual(4, first.table_rows[0].unresolved_lights)
        self.assertEqual(3, second.table_rows[0].unresolved_lights)
        self.assertEqual(4, second.table_rows[1].lights)
        for row in second.table_rows:
            self.assertEqual(("B0",), row.seatings["A0"])
        self.assertTrue(all(row.lights is None and row.unresolved_lights is None for row in second.table_rows[2:]))

    def test_late_matchbox_changes_old_status_but_not_old_seating(self):
        data = toy_data(
            ["A", "B", "C"], ["x", "y", "z"],
            weeks=[week(1, night([["A", "x"], ["B", "y"], ["C", "z"]], 1), box("A", "y", "yes"))],
        )
        snapshot = build_villa_snapshot(data, 1)
        self.assertEqual(("x",), snapshot.table_rows[0].seatings["A"])
        self.assertEqual(("y",), snapshot.table_rows[1].seatings["A"])
        self.assertEqual(("z",), snapshot.table_rows[1].seatings["C"])
        self.assertEqual("no", snapshot.pair_states[("A", "x")].result)
        self.assertEqual("matching_night", snapshot.pair_states[("C", "z")].source)
        self.assertEqual(0, snapshot.table_rows[0].unresolved_lights)
        self.assertEqual("0", snapshot.table_rows[0].remaining_label)

    def test_later_confirmation_does_not_add_a_missing_past_seat(self):
        data = toy_data(
            ["A", "B", "C"], ["x", "y", "z"],
            weeks=[
                week(1, night([["A", "x"], ["B", "y"]], 1)),
                week(2, box("A", "x", "no"), box("C", "z", "yes"), night([], 2, automatic_lights=2)),
            ],
        )
        snapshot = build_villa_snapshot(data, 2)
        self.assertNotIn("C", snapshot.table_rows[0].seatings)
        self.assertEqual(("z",), snapshot.table_rows[1].seatings["C"])
        self.assertEqual(("y",), snapshot.table_rows[1].seatings["B"])
        self.assertEqual(0, snapshot.table_rows[0].unresolved_lights)

    def test_double_match_has_both_women_and_shared_partner_in_center(self):
        data = toy_data(
            ["A", "B"], ["x", "y", "z"],
            multi=[{"group": "bits", "size": 2}],
            weeks=[week(1, box(
                "A", "x", "yes",
                revealed_group={"partner": "A", "members": ["x", "y"], "complete": True},
            ), night([["B", "z"]], 2, automatic_lights=1))],
        )
        snapshot = build_villa_snapshot(data, 1)
        self.assertEqual(("x", "A", "y"), snapshot.matchboxes[0].people)
        self.assertTrue(all(row.seatings["A"] == ("x", "y") for row in snapshot.table_rows))
        self.assertEqual(0, snapshot.table_rows[0].unresolved_lights)
        self.assertEqual(2, snapshot.table_rows[0].lights)

    def test_remaining_fractions_update_without_future_information(self):
        data = json.loads(Path("ayto_data.json").read_text(encoding="utf-8"))["s6vip"]
        first = build_villa_snapshot(data, 1)
        third = build_villa_snapshot(data, 3)
        self.assertEqual("3/10", first.table_rows[0].remaining_label)
        self.assertEqual("2/7", third.table_rows[0].remaining_label)
        self.assertEqual(7, third.table_rows[0].open_pair_count)
        self.assertTrue(all(row.remaining_label is None for row in third.table_rows[3:]))

    def test_sold_box_and_night_stay_unknown(self):
        snapshot = build_villa_snapshot(ten_data(week(
            1, box("A0", "B0", "sold"), night(ten_pairs(), result="sold"),
        )), 1)
        self.assertEqual("sold", snapshot.matchboxes[0].result)
        self.assertIsNone(snapshot.table_rows[0].lights)
        self.assertIsNone(snapshot.table_rows[0].unresolved_lights)
        self.assertEqual("?/10", snapshot.table_rows[0].remaining_label)
        self.assertTrue(snapshot.table_rows[0].played)
        self.assertFalse(snapshot.table_rows[1].played)
        self.assertFalse(snapshot.table_rows[1].seatings)

    def test_current_season_counts_and_prefilled_double_match(self):
        data = json.loads(Path("ayto_data.json").read_text(encoding="utf-8"))["s6vip"]
        snapshot = build_villa_snapshot(data, 8)
        self.assertEqual([3, 3, 3, 4, 3, 5, 5, 6], [row.lights for row in snapshot.table_rows[:8]])
        self.assertEqual([2, 1, 2, 2, 1, 3, 3, 4], [row.unresolved_lights for row in snapshot.table_rows[:8]])
        for row in snapshot.table_rows[2:]:
            self.assertEqual({"Marta", "Janice"}, set(row.seatings["Johannes"]))
        for row in snapshot.table_rows[6:]:
            self.assertEqual(("Jenny",), row.seatings["Daymian"])
        self.assertNotIn("Marwin", snapshot.table_rows[7].seatings)
        self.assertEqual("sold", snapshot.matchboxes[-1].result)


class VillaRendererTests(unittest.TestCase):
    def artwork(self, root, data):
        background, faces = root / "background.png", root / "faces.png"
        Image.new("RGB", POST_SIZE, "#111111").save(background)
        Image.new("RGBA", POST_SIZE, (0, 0, 0, 0)).save(faces)
        data["style"] = {"background": str(background), "face_layer": str(faces)}
        return data

    def test_labels_fit_dense_grid_in_both_layouts(self):
        for columns, rows in ((10, 10), (10, 11), (11, 11), (10, 12)):
            _, _, width, height = heatmap_bounds(columns, rows)
            for night_rows in (3, 4):
                for label in ("4/10", "3/9", "10/11", "?"):
                    with self.subTest(columns=columns, rows=rows, night_rows=night_rows, label=label):
                        slot_w, slot_h = width / columns / 3, height / rows / night_rows
                        font = _villa_label_font(label, slot_w, slot_h)
                        left, top, right, bottom = font.getbbox(label)
                        self.assertLessEqual(right - left, slot_w - 3)
                        self.assertLessEqual(bottom - top, slot_h - 3)

    def test_portrait_overlays_do_not_obscure_labels_or_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.artwork(root, ten_data(
                *(week(i, night(ten_pairs(), 4)) for i in range(1, 11))
            ))
            data["weeks"][0]["events"].insert(0, box("A0", "B0", "no"))
            data["weeks"][-1]["events"].append(box("A1", "B2", "yes"))
            face = Image.new("RGBA", POST_SIZE, (0, 0, 0, 0))
            face_draw = ImageDraw.Draw(face)
            x, y, width, height = heatmap_bounds(10, 10)
            face_draw.rectangle(
                (x, y, x + 17, y + height), fill=(255, 0, 255, 255)
            )
            face_draw.rectangle(
                (x, y + height - 12, x + width, y + height),
                fill=(255, 0, 255, 255),
            )
            face.save(data["style"]["face_layer"])
            alpha = face.getchannel("A")
            with patch(
                "insta_renderer._draw_centered_text", wraps=insta_renderer._draw_centered_text
            ) as draw_text:
                render_villa_post("toy", data, build_villa_snapshot(data, 10), root)
            for call in draw_text.call_args_list:
                _, (cx, cy), text, font, _ = call.args
                left, top, right, bottom = font.getbbox(
                    text, stroke_width=call.kwargs.get("stroke_width", 0)
                )
                bounds = (
                    int(np.floor(cx - (right - left) / 2)),
                    int(np.floor(cy - (bottom - top) / 2)),
                    int(np.ceil(cx + (right - left) / 2)),
                    int(np.ceil(cy + (bottom - top) / 2)),
                )
                self.assertIsNone(alpha.crop(bounds).getbbox(), (text, bounds))

    def test_night_slots_symbols_uniform_color_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = ten_pairs()
            data = self.artwork(root, ten_data(
                *(week(i, night(pairs, 4)) for i in range(1, 11))
            ))
            data["weeks"][0]["events"].insert(0, box("A0", "B0", "no"))
            data["weeks"][-1]["events"].append(box("A1", "B2", "yes"))
            snapshot = build_villa_snapshot(data, 10)
            with patch(
                "insta_renderer._draw_centered_text", wraps=insta_renderer._draw_centered_text
            ) as draw_text:
                path = render_villa_post("toy", data, snapshot, root)
            self.assertEqual("toy_10_insta_Villa.png", path.name)
            with Image.open(path) as image:
                self.assertEqual(POST_SIZE, image.size)
                self.assertEqual(ImageColor.getrgb(VILLA_CELL_COLOR), image.getpixel((225, 275)))
            calls = [call.args for call in draw_text.call_args_list]
            symbols = [call for call in calls if call[4] in ("red", "green")]
            self.assertEqual(1, sum(call[4] == "green" for call in symbols))
            red_sizes = [call[3].size for call in symbols if call[4] == "red"]
            self.assertEqual(2, len(set(red_sizes)))
            x0, y0, width, height = heatmap_bounds(10, 10)
            cell = width / 10
            # Pair A3/B3 remains open in every night; MN 10 occupies row 4, col 1.
            expected = (x0 + 3 * cell + cell / 6, y0 + 3 * cell + cell * 7 / 8)
            self.assertTrue(any(
                call[2] == "4/7" and call[1] == expected for call in calls
            ))
            self.assertEqual(7 * 10, sum(call[2] == "4/7" for call in calls))
            self.assertEqual({path.name}, {p.name for p in root.glob("toy*.png")})


class VillaTableRendererTests(unittest.TestCase):
    def artwork(self, root, data):
        VillaRendererTests().artwork(root, data)
        directory = root / "portraits"
        directory.mkdir()
        from ayto_villa import villa_portrait_names
        for index, name in enumerate(villa_portrait_names(data)):
            face = Image.new("RGBA", (90, 90), (0, 0, 0, 0))
            ImageDraw.Draw(face).ellipse((8, 2, 82, 88), fill=(60, 100 + index * 3, 210, 255))
            face.save(directory / f"{name}.png")
        data["style"]["villa_faces"] = str(directory)
        return data

    def test_table_portraits_markers_and_future_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.artwork(root, toy_data(
                ["A", "B", "C"], ["x", "y", "z"],
                weeks=[week(1, night([["A", "x"], ["B", "y"], ["C", "z"]], 1), box("A", "y", "yes"))],
            ))
            snapshot = build_villa_snapshot(data, 1)
            path = render_villa_post("toy", data, snapshot, root)
            self.assertEqual("toy_1_insta_Villa.png", path.name)
            x_edges, y_edges = villa_table_bounds(3)
            with Image.open(path) as image:
                self.assertEqual(POST_SIZE, image.size)
                # Historical A/x is excluded; future A/y is a known match.
                first = np.asarray(image.crop((x_edges[1], y_edges[1], x_edges[2], y_edges[2])))
                future = np.asarray(image.crop((x_edges[1], y_edges[2], x_edges[2], y_edges[3])))
                self.assertTrue(np.any(np.all(first == (255, 0, 0), axis=2)))
                self.assertTrue(np.any(np.all(future == (8, 72, 8), axis=2)))
                # B has no confirmed match, so its future slot stays empty.
                center = ((x_edges[2] + x_edges[3]) // 2, (y_edges[2] + y_edges[3]) // 2)
                self.assertEqual((131, 130, 127), image.getpixel(center))
                self.assertEqual((17, 17, 17), image.getpixel((center[0], center[1] + 10)))
                self.assertEqual((131, 130, 127), image.getpixel((center[0] + 10, center[1])))
                self.assertEqual((0, 0, 0), image.getpixel((x_edges[0] + 1, y_edges[1])))
                # Future light totals have white discs without numbers.
                center = ((x_edges[-3] + x_edges[-2]) // 2, (y_edges[-2] + y_edges[-1]) // 2)
                self.assertEqual((131, 130, 127), image.getpixel(center))
                # The header connector stops inside the allocation disc.
                header_y = (y_edges[0] + y_edges[1]) // 2
                remaining_x = (x_edges[-2] + x_edges[-1]) // 2
                self.assertEqual((17, 17, 17), image.getpixel((remaining_x + 29, header_y)))

    def test_double_match_portraits_and_ten_box_cards_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.artwork(root, toy_data(
                ["A", "B"], ["x", "y", "z"],
                multi=[{"group": "bits", "size": 2}],
                weeks=[week(1, box("A", "x", "yes", revealed_group={
                    "partner": "A", "members": ["x", "y"], "complete": True,
                }), *(box("B", "z", "sold") for _ in range(9)), night([["B", "z"]], 2, automatic_lights=1))],
            ))
            with patch("insta_renderer._paste_villa_portrait", wraps=insta_renderer._paste_villa_portrait) as paste:
                path = render_villa_post("toy", data, build_villa_snapshot(data, 1), root)
            for call in paste.call_args_list:
                left, top, right, bottom = call.args[2]
                self.assertTrue(0 <= left < right <= POST_SIZE[0])
                self.assertTrue(210 <= top < bottom <= 1290)
            with Image.open(path) as image:
                self.assertEqual(POST_SIZE, image.size)

    def test_markers_letter_headers_and_compact_matchbox_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.artwork(root, toy_data(
                ["A", "B", "C"], ["x", "y", "z"],
                weeks=[week(1, night([["A", "x"], ["B", "y"], ["C", "z"]], 1), box("A", "y", "yes"), box("B", "x", "sold"))],
            ))
            with patch("insta_renderer._draw_centered_text", wraps=insta_renderer._draw_centered_text) as text, patch(
                "insta_renderer._draw_villa_marker", wraps=insta_renderer._draw_villa_marker,
            ) as marker:
                render_villa_post("toy", data, build_villa_snapshot(data, 1), root)
            labels = [call.args[2] for call in text.call_args_list]
            self.assertIn("W", labels)
            self.assertIn("L", labels)
            self.assertIn("?", labels)
            for label, color in (("L", "yellow"), ("?", "red")):
                header = next(call for call in text.call_args_list if call.args[2] == label)
                self.assertEqual(color, header.args[4])
                self.assertEqual(2, header.kwargs["stroke_width"])
            self.assertIn("W1", labels)
            self.assertNotIn("#", labels)
            self.assertFalse(any(label.startswith("MB ") or "verkauft" in label for label in labels))
            self.assertNotIn("Lichter", labels)
            self.assertNotIn("offen", labels)
            self.assertFalse(any(label.startswith("Leere MN-Zeilen") or label.startswith("Lichter gesamt") for label in labels))
            self.assertTrue(all(call.args[3] >= 45 for call in marker.call_args_list))
            self.assertTrue(any(call.args[2] == "yes" and call.args[3] == 45 for call in marker.call_args_list))
            self.assertTrue(any(call.args[2] == "no" and call.args[3] == 54 for call in marker.call_args_list))
            self.assertEqual(60, marker.call_args_list[-1].args[3])
            self.assertTrue(any(call.args[2] == "$" and call.args[4] == "#66ccff" for call in text.call_args_list))
            sold_font = next(call.args[3] for call in text.call_args_list if call.args[2] == "$")
            sold_ink = sold_font.getbbox("$", stroke_width=2)
            cross_ink = insta_renderer._font(60).getbbox("\u00d7", stroke_width=2)
            plus_ink = insta_renderer._font(60).getbbox("+", stroke_width=2)
            self.assertLessEqual(sold_ink[3] - sold_ink[1], max(cross_ink[3] - cross_ink[1], plus_ink[3] - plus_ink[1]))

    def test_overlap_and_antialiased_allocation_capsule(self):
        image = Image.new("RGB", POST_SIZE, "#111111")
        portraits = {name: Image.new("RGBA", (70, 90), "white") for name in ("A", "B", "C")}
        union, placed = insta_renderer._paste_villa_group(image, portraits, ("A", "B", "C"), (100, 100, 290, 190), overlap=0.4)
        self.assertLess(placed[1][0], placed[0][2])
        self.assertLess(placed[2][0], placed[1][2])
        self.assertLess(union[2] - union[0], sum(bounds[2] - bounds[0] for bounds in placed))
        draw = ImageDraw.Draw(image)
        for label in ("3/10", "2/7", "$/10", "0"):
            with patch("insta_renderer._draw_centered_text", wraps=insta_renderer._draw_centered_text) as text:
                insta_renderer._draw_villa_capsule(image, (600, 600), 24, label, color="red")
            call = text.call_args
            font = call.args[3]
            self.assertEqual(26, font.size)
            self.assertEqual((600, 600), call.args[1])
            left, top, right, bottom = font.getbbox(label, stroke_width=2)
            self.assertLessEqual(right - left, 88)
            self.assertLessEqual(bottom - top, 48)
            paper = ImageColor.getrgb(VILLA_PAPER_COLOR)
            self.assertEqual(paper, image.getpixel((559, 600)))
            self.assertEqual(paper, image.getpixel((641, 600)))
            self.assertEqual(paper, image.getpixel((600, 580)))
        # Curved edges have blended pixels instead of a hard staircase.
        edge = np.asarray(image.crop((554, 575, 560, 625)))
        self.assertTrue(np.any((edge[:, :, 0] > 17) & (edge[:, :, 0] < paper[0])))

    def test_portraits_are_checked_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.artwork(root, ten_data(week(1, night(ten_pairs(), 4))))
            self.assertEqual([], validate_artwork("toy", data))
            (Path(data["style"]["villa_faces"]) / "A0.png").unlink()
            with self.assertRaisesRegex(DataValidationError, "Villa portrait not found"):
                validate_artwork("toy", data)


class VillaExportTests(unittest.TestCase):
    def test_checkpoint_backfill_cutoff_and_stable_previous_week(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            season = str(root / "toy")
            output = Path(season)
            output.mkdir()
            existing = output / "existing.png"
            existing.write_bytes(b"existing post")
            data = ten_data(
                week(1, night(ten_pairs(), 4)),
                week(2, box("A0", "B0", "no"), night(ten_pairs(), 4)),
                week(3, box("A1", "B1", "sold")),
            )
            VillaTableRendererTests().artwork(root, data)
            data["status"] = "completed"
            checkpoint = SimpleNamespace(state=SimpleNamespace(
                processed_events=len(flatten_events(data)), completed_week=3, night_records=[],
            ))
            with patch("aytorakel.load_checkpoint", return_value=checkpoint), patch(
                "aytorakel.SeasonSolver"
            ) as solver, patch("aytorakel.render_summary", return_value=None), patch(
                "aytorakel._render_new_snapshots"
            ) as old_posts:
                run_season(season, data, prevalidated=True, through_week=1)
                self.assertEqual(1, len(list(output.glob("*_insta_Villa.png"))))
                first = (output / "toy_1_insta_Villa.png").read_bytes()
                run_season(season, data, prevalidated=True)
                self.assertEqual(
                    {"toy_1_insta_Villa.png", "toy_2_insta_Villa.png"},
                    {p.name for p in output.glob("*_insta_Villa.png")},
                )
                self.assertEqual(first, (output / "toy_1_insta_Villa.png").read_bytes())
                self.assertEqual(b"existing post", existing.read_bytes())
                solver.assert_not_called()
                old_posts.assert_not_called()

    def test_normal_export_keeps_old_posts_and_villa_survives_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            season = str(root / "toy")
            data = toy_data(
                ["A", "B"], ["x", "y"],
                weeks=[week(1, night([["A", "x"], ["B", "y"]], 2))],
            )
            VillaRendererTests().artwork(root, data)
            data["status"] = "completed"
            with patch("aytorakel.render_cover", return_value=root / "cover.png") as cover, patch(
                "aytorakel._render_new_snapshots", return_value=[]
            ) as old_posts, patch("aytorakel.save_checkpoint", return_value=root / "check.npz"), patch(
                "aytorakel.render_summary", return_value=None
            ):
                run_season(season, data, prevalidated=True, from_scratch=True)
                cover.assert_called_once()
                old_posts.assert_called_once()
                self.assertTrue((Path(season) / "toy_1_insta_Villa.png").is_file())


if __name__ == "__main__":
    unittest.main()

