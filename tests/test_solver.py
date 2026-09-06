import unittest

import numpy as np

from ayto_solver import (
    SeasonSolver,
    _multi_match_member_mask,
    _probabilities,
    apply_evidence_event,
    faster_permutations,
    generate_solution_batches,
    matching_night_scores,
    mask_dtype,
    validate_season_data,
)


def collect(rows, bits, rules):
    return np.concatenate(list(generate_solution_batches(rows, bits, rules, batch_size=2)))


def brute_force_graphs(rows, bits, rules):
    """Independent edge-subset enumeration used only for tiny fixtures."""
    expected_row_degrees = [1] * len(rows)
    expected_bit_degrees = [1] * len(bits)
    for rule in rules:
        degrees = expected_row_degrees if rule["group"] == "bits" else expected_bit_degrees
        degrees.remove(1)
        degrees.append(int(rule["size"]))
    expected_row_degrees.sort()
    expected_bit_degrees.sort()

    graphs = set()
    for edges in range(1 << (len(rows) * len(bits))):
        masks = tuple(
            sum(
                1 << bit
                for bit in range(len(bits))
                if edges & (1 << (row * len(bits) + bit))
            )
            for row in range(len(rows))
        )
        row_degrees = sorted(int(mask).bit_count() for mask in masks)
        bit_degrees = sorted(
            sum(bool(mask & (1 << bit)) for mask in masks)
            for bit in range(len(bits))
        )
        if row_degrees == expected_row_degrees and bit_degrees == expected_bit_degrees:
            graphs.add(masks)
    return graphs


def toy_data(rows, bits, multi=None, weeks=None):
    return {
        "schema_version": 2,
        "initial_active": "all",
        "groups": {"a": list(rows), "b": list(bits)},
        "matrix": {"rows": "a", "bits": "b"},
        "initial_match_model": {"multi_matches": list(multi or [])},
        "style": {"background": "background.png", "face_layer": "faces.png"},
        "weeks": list(weeks or []),
    }


class GeneratorTests(unittest.TestCase):
    def test_fast_permutations_are_complete_and_unique(self):
        permutations = faster_permutations(4)
        self.assertEqual((24, 4), permutations.shape)
        self.assertEqual(24, len({tuple(row) for row in permutations}))
        self.assertTrue(all(sorted(row) == [0, 1, 2, 3] for row in permutations.tolist()))

    def test_mask_dtype_grows_after_sixteen_bits(self):
        self.assertEqual(np.dtype(np.uint16), mask_dtype(16))
        self.assertEqual(np.dtype(np.uint32), mask_dtype(17))

    def test_normal_double_triple_and_two_doubles_counts(self):
        fixtures = [
            (["A", "B"], ["x", "y"], [], 2),
            (
                ["A", "B"], ["x", "y", "z"],
                [{"group": "bits", "size": 2, "known_members": []}], 6,
            ),
            (
                ["A", "B"], ["w", "x", "y", "z"],
                [{"group": "bits", "size": 3, "known_members": []}], 8,
            ),
            (
                ["A", "B"], ["w", "x", "y", "z"],
                [
                    {"group": "bits", "size": 2, "known_members": []},
                    {"group": "bits", "size": 2, "known_members": []},
                ],
                6,
            ),
        ]
        for rows, bits, rules, expected_count in fixtures:
            solutions = collect(rows, bits, rules)
            self.assertEqual(expected_count, len(solutions))
            self.assertEqual(len(solutions), len({tuple(row) for row in solutions}))
            self.assertEqual(
                brute_force_graphs(rows, bits, rules),
                {tuple(int(value) for value in row) for row in solutions},
            )

    def test_row_side_double_uses_same_fixed_representation(self):
        solutions = collect(
            ["A", "B", "C"], ["x", "y"],
            [{"group": "rows", "size": 2, "known_members": []}],
        )
        self.assertEqual(6, len(solutions))
        self.assertEqual(6, len({tuple(row) for row in solutions}))
        self.assertTrue(all(np.bitwise_count(row).sum() == 3 for row in solutions))
        self.assertEqual(
            brute_force_graphs(
                ["A", "B", "C"], ["x", "y"],
                [{"group": "rows", "size": 2, "known_members": []}],
            ),
            {tuple(int(value) for value in row) for row in solutions},
        )

    def test_canonical_edge_probabilities_do_not_weight_seating(self):
        solutions = collect(
            ["A", "B"], ["x", "y", "z"],
            [{"group": "bits", "size": 2, "known_members": []}],
        )
        probabilities = _probabilities(solutions, 3)
        np.testing.assert_allclose(probabilities, np.full((2, 3), 50.0))
        np.testing.assert_allclose(probabilities.sum(axis=0), np.full(3, 100.0))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.full(2, 150.0))


class EventTests(unittest.TestCase):
    def test_matching_night_counts_edges_once(self):
        data = toy_data(["A", "B"], ["x", "y", "z"])
        solver = SeasonSolver("toy", data)
        solutions = np.asarray([[3, 4]], dtype=np.uint16)  # A-x/y, B-z
        event = {"type": "matching_night", "pairs": [["A", "y"], ["B", "z"]], "lights": 2}
        np.testing.assert_array_equal(matching_night_scores(solver.state, solutions, event), [2])
        filtered, _ = apply_evidence_event(solver.state, solutions, event)
        self.assertEqual(1, len(filtered))

    def test_automatic_lights_are_included(self):
        data = toy_data(["A", "B"], ["x", "y"])
        solver = SeasonSolver("toy", data)
        solutions = collect(["A", "B"], ["x", "y"], [])
        event = {
            "type": "matching_night",
            "pairs": [["A", "x"]],
            "automatic_lights": 1,
            "lights": 2,
        }
        np.testing.assert_array_equal(
            matching_night_scores(solver.state, solutions, event), [2, 1]
        )

    def test_match_box_yes_no_and_sold(self):
        data = toy_data(["A", "B"], ["x", "y"])
        solver = SeasonSolver("toy", data)
        solutions = collect(["A", "B"], ["x", "y"], [])
        yes, _ = apply_evidence_event(solver.state, solutions, {"type": "box", "pair": ["A", "x"], "result": "yes"})
        no, _ = apply_evidence_event(solver.state, solutions, {"type": "box", "pair": ["A", "x"], "result": "no"})
        sold, _ = apply_evidence_event(solver.state, solutions, {"type": "box", "pair": ["A", "x"], "result": "sold"})
        self.assertEqual(1, len(yes))
        self.assertEqual(1, len(no))
        self.assertEqual(2, len(sold))

    def test_normal_perfect_match_excludes_both_people_from_multi_match(self):
        data = toy_data(["Dino", "Joshua", "Levin", "Tano"], ["Deisy", "Sophia", "x", "y", "z"])
        solver = SeasonSolver("toy", data)
        solutions = np.asarray(
            [
                [3, 4, 8, 16],   # Sophia is additionally matched with Dino
                [1, 6, 8, 16],   # Sophia is additionally matched with Joshua
                [1, 4, 10, 16],  # Sophia is additionally matched with Levin
                [1, 4, 8, 18],   # Sophia is additionally matched with Tano
            ],
            dtype=np.uint16,
        )
        filtered, _ = apply_evidence_event(
            solver.state,
            solutions,
            {"type": "box", "pair": ["Dino", "Deisy"], "result": "yes"},
        )
        self.assertEqual(3, len(filtered))
        self.assertTrue(all(int(row[0]) == 1 for row in filtered))

    def test_unknown_arrival_rebuilds_and_replays_old_evidence(self):
        weeks = [
            {"number": 1, "events": [{"type": "box", "pair": ["A", "x"], "result": "no"}]},
            {
                "number": 2,
                "events": [{
                    "type": "cast_change",
                    "add": [{"group": "b", "person": "z"}],
                    "face_layer": "faces2.png",
                    "match_update": {
                        "mode": "rebuild",
                        "multi_matches": [{"group": "bits", "size": 2, "known_members": []}],
                    },
                }],
            },
        ]
        solver = SeasonSolver("toy", toy_data(["A", "B"], ["x", "y"], weeks=weeks))
        solver.process_week(weeks[0], 0)
        solver.process_week(weeks[1], 1)
        self.assertEqual(3, len(solver.state.solutions))
        self.assertTrue(all((row[0] & 1) == 0 for row in solver.state.solutions))

    def test_extend_supports_new_people_on_both_sides(self):
        bit_add = {
            "number": 2,
            "events": [{
                "type": "cast_change",
                "add": [{"group": "b", "person": "z"}],
                "match_update": {
                    "mode": "extend",
                    "new_multi_match": {"group": "bits", "size": 2, "known_members": ["z"]},
                },
            }],
        }
        data = toy_data(["A", "B"], ["x", "y"], weeks=[{"number": 1, "events": []}, bit_add])
        solver = SeasonSolver("toy", data)
        solver.process_week(data["weeks"][0], 0)
        solver.process_week(bit_add, 0)
        self.assertEqual(4, len(solver.state.solutions))

        row_add = {
            "number": 2,
            "events": [{
                "type": "cast_change",
                "add": [{"group": "a", "person": "C"}],
                "match_update": {
                    "mode": "extend",
                    "new_multi_match": {"group": "rows", "size": 2, "known_members": ["C"]},
                },
            }],
        }
        data = toy_data(["A", "B"], ["x", "y"], weeks=[{"number": 1, "events": []}, row_add])
        solver = SeasonSolver("toy", data)
        solver.process_week(data["weeks"][0], 0)
        solver.process_week(row_add, 0)
        self.assertEqual(4, len(solver.state.solutions))
        self.assertTrue(all(len(row) == 3 for row in solver.state.solutions))

    def test_extend_inserts_new_people_alphabetically_and_remaps_graphs(self):
        from ayto_solver import _expand_with_new_participants

        data = toy_data(["B", "D"], ["x", "z"])
        solver = SeasonSolver("toy", data)
        solver.state.solutions = np.asarray([[1, 2], [2, 1]], dtype=np.uint16)
        bit_event = {
            "type": "cast_change",
            "add": [{"group": "b", "person": "y"}],
            "match_update": {
                "mode": "extend",
                "eligible_partners": ["B"],
                "new_multi_match": {"group": "bits", "size": 2, "known_members": ["y"]},
            },
        }
        old_rows, old_bits = solver._apply_cast_state(bit_event)
        expanded = _expand_with_new_participants(
            solver.state, solver.state.solutions, bit_event, old_rows, old_bits
        )
        self.assertEqual(["x", "y", "z"], solver.state.bit_names)
        np.testing.assert_array_equal(expanded, [[3, 4], [6, 1]])

        data = toy_data(["B", "D"], ["x", "z"])
        solver = SeasonSolver("toy", data)
        solver.state.solutions = np.asarray([[1, 2], [2, 1]], dtype=np.uint16)
        row_event = {
            "type": "cast_change",
            "add": [{"group": "a", "person": "C"}],
            "match_update": {
                "mode": "extend",
                "eligible_partners": ["x"],
                "new_multi_match": {"group": "rows", "size": 2, "known_members": ["C"]},
            },
        }
        old_rows, old_bits = solver._apply_cast_state(row_event)
        expanded = _expand_with_new_participants(
            solver.state, solver.state.solutions, row_event, old_rows, old_bits
        )
        self.assertEqual(["B", "C", "D"], solver.state.row_names)
        np.testing.assert_array_equal(expanded, [[1, 1, 2], [2, 1, 1]])

    def test_johannes_laurenz_shape_needs_no_orientation_change(self):
        data = toy_data(["Johannes", "Other"], ["Marta", "Janice", "Zoe"])
        solver = SeasonSolver("toy", data)
        solver.state.solutions = np.asarray([[3, 4]], dtype=np.uint16)
        event = {
            "type": "cast_change",
            "add": [{"group": "a", "person": "Laurenz"}],
            "remove": ["Johannes", "Marta", "Janice"],
            "match_update": {
                "mode": "extend",
                "eligible_partners": ["Zoe"],
                "new_multi_match": {"group": "rows", "size": 2, "known_members": ["Laurenz"]},
            },
        }
        old_rows, old_bits = solver._apply_cast_state(event)
        from ayto_solver import _expand_with_new_participants

        expanded = _expand_with_new_participants(solver.state, solver.state.solutions, event, old_rows, old_bits)
        np.testing.assert_array_equal(expanded, [[3, 4, 4]])

    def test_multi_match_members_work_on_both_sides(self):
        data = toy_data(["A", "B", "C"], ["x", "y", "z"])
        solver = SeasonSolver("toy", data)
        solutions = np.asarray(
            [
                [3, 4, 4],  # A has x/y; B and C share z
                [1, 2, 4],  # all normal
            ],
            dtype=np.uint16,
        )
        np.testing.assert_array_equal(
            _multi_match_member_mask(solver.state, solutions, "y"), [True, False]
        )
        np.testing.assert_array_equal(
            _multi_match_member_mask(solver.state, solutions, "C"), [True, False]
        )

    def test_complete_reveal_removal_and_solution(self):
        data = toy_data(["A", "B"], ["x", "y", "z"])
        solver = SeasonSolver("toy", data)
        solutions = collect(
            ["A", "B"],
            ["x", "y", "z"],
            [{"group": "bits", "size": 2, "known_members": []}],
        )
        reveal = {
            "type": "reveal",
            "complete_groups": [
                {"partner": "A", "members": ["x", "y"], "complete": True}
            ],
        }
        revealed, _ = apply_evidence_event(solver.state, solutions, reveal)
        self.assertEqual(1, len(revealed))
        solver.state.solutions = revealed
        solver.state.active["a"].remove("A")
        solver.state.active["b"].difference_update({"x", "y"})
        solved, _ = apply_evidence_event(
            solver.state,
            solver.state.solutions,
            {"type": "solution", "pairs": [["A", "x"], ["A", "y"], ["B", "z"]]},
        )
        self.assertEqual(1, len(solved))


class ValidationTests(unittest.TestCase):
    def test_toy_schema_validates(self):
        data = toy_data(
            ["A", "B"], ["x", "y"], weeks=[{
                "number": 1,
                "events": [{"type": "matching_night", "lights": 1, "pairs": [["A", "x"]]}],
            }]
        )
        validate_season_data("toy", data)

    def test_incompatible_topology_is_rejected(self):
        data = toy_data(["A", "B"], ["x", "y", "z"])
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_season_data("toy", data)


if __name__ == "__main__":
    unittest.main()
