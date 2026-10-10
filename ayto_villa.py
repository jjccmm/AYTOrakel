"""Villa knowledge derived from public events, without enumerating solutions.

Each week is rebuilt from its own event prefix. Only direct revelations,
partner capacity limits, and completely resolved Matching Nights are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ayto_solver import DataValidationError, flatten_events


Pair = tuple[str, str]


@dataclass(frozen=True)
class VillaPairState:
    result: str
    source: str


@dataclass(frozen=True)
class VillaNight:
    number: int
    week: int
    pairs: tuple[Pair, ...]
    unresolved_lights: int | None
    open_pairs: tuple[Pair, ...]
    lights: int | None = None
    automatic_lights: int = 0

    @property
    def label(self) -> str | None:
        if not self.open_pairs:
            return None
        if self.unresolved_lights is None:
            return "?"
        return f"{self.unresolved_lights}/{len(self.open_pairs)}"


@dataclass(frozen=True)
class VillaTableRow:
    number: int
    week: int | None
    seatings: dict[str, tuple[str, ...]]
    lights: int | None
    unresolved_lights: int | None
    played: bool
    open_pair_count: int | None = None

    @property
    def remaining_label(self) -> str | None:
        if not self.played:
            return None
        if self.unresolved_lights is None:
            return f"?/{self.open_pair_count}" if self.open_pair_count else "?"
        if not self.open_pair_count:
            # A fully solved night has no allocation left; avoid a 0/0 fraction.
            return str(self.unresolved_lights)
        return f"{self.unresolved_lights}/{self.open_pair_count}"


@dataclass(frozen=True)
class VillaMatchBox:
    number: int
    week: int
    people: tuple[str, ...]
    result: str


@dataclass(frozen=True)
class VillaSnapshot:
    week: int
    row_names: tuple[str, ...]
    bit_names: tuple[str, ...]
    face_layer: str
    pair_states: dict[Pair, VillaPairState]
    nights: tuple[VillaNight, ...]
    table_columns: tuple[str, ...] = ()
    table_rows: tuple[VillaTableRow, ...] = ()
    matchboxes: tuple[VillaMatchBox, ...] = ()

    @property
    def night_rows(self) -> int:
        return max(3, (len(self.nights) + 2) // 3)


def _derive_villa_knowledge(
    data: dict[str, Any], week: int, history: list[tuple[int, int, dict[str, Any]]]
) -> VillaSnapshot:
    """Reduce a public event prefix, without consulting any solver solutions."""
    row_group, bit_group = data["matrix"]["rows"], data["matrix"]["bits"]
    groups = {key: list(names) for key, names in data["groups"].items()}
    initial_active = data["initial_active"]
    active = {
        key: set(names if initial_active == "all" else initial_active[key])
        for key, names in groups.items()
    }
    face_layer = data["style"]["face_layer"]
    limits = {name: 1 for names in groups.values() for name in names}
    arrival_exclusions: list[tuple[str, str]] = []

    def apply_model(rules: list[dict[str, Any]]) -> None:
        for rule in rules:
            # The rule's group contains the members, not their shared partner.
            partner_group = row_group if rule["group"] == "bits" else bit_group
            for person in groups[partner_group]:
                limits[person] = max(limits[person], int(rule["size"]))

    apply_model(data["initial_match_model"].get("multi_matches", []))
    for _, _, event in history:
        if event["type"] != "cast_change":
            continue
        for addition in event.get("add", []):
            group, person = addition["group"], addition["person"]
            groups[group].append(person)
            groups[group].sort(key=str.casefold)
            active[group].add(person)
            limits[person] = 1
        for person in event.get("remove", []):
            for names in active.values():
                names.discard(person)
        face_layer = event.get("face_layer", face_layer)
        update = event.get("match_update", {})
        if update.get("mode") == "rebuild":
            limits = {name: 1 for names in groups.values() for name in names}
            arrival_exclusions.clear()
            apply_model(update.get("multi_matches", []))
        elif update.get("mode") == "extend":
            addition = event["add"][0]
            other_group = bit_group if addition["group"] == row_group else row_group
            eligible = set(update.get("eligible_partners", active[other_group]))
            size = int(update["new_multi_match"]["size"])
            for person in eligible:
                limits[person] = max(limits[person], size)
            arrival_exclusions.extend(
                (addition["person"], person)
                for person in groups[other_group]
                if person not in eligible
            )

    row_names, bit_names = tuple(groups[row_group]), tuple(groups[bit_group])
    row_set, bit_set = set(row_names), set(bit_names)
    states: dict[Pair, VillaPairState] = {}
    closed: dict[str, set[str]] = {}

    def canonical(pair: list[str] | tuple[str, str]) -> Pair:
        a, b = pair
        if a in row_set and b in bit_set:
            return a, b
        if b in row_set and a in bit_set:
            return b, a
        raise DataValidationError(f"Villa W{week}: invalid pair {pair!r}")

    def mark(pair: Pair, result: str, source: str) -> bool:
        previous = states.get(pair)
        if previous is not None:
            if previous.result != result:
                raise DataValidationError(
                    f"Villa W{week}: contradictory information for {pair!r}"
                )
            return False
        states[pair] = VillaPairState(result, source)
        return True

    def close(person: str, partners: set[str]) -> None:
        if person in closed and closed[person] != partners:
            raise DataValidationError(
                f"Villa W{week}: contradictory complete match for {person}"
            )
        closed[person] = partners

    def reveal_group(group: dict[str, Any]) -> None:
        partner, members = group["partner"], set(group["members"])
        for member in group["members"]:
            mark(canonical((partner, member)), "yes", "reveal")
        if group.get("complete", True):
            close(partner, members)

    # Seed explicit results first: direct No Matches retain their large X.
    for _, _, event in history:
        event_type = event["type"]
        if event_type == "box":
            result = event["result"]
            if result != "sold":
                pair = canonical(event["pair"])
                mark(pair, result, "box")
                if result == "yes" and event.get(
                    "complete_pair", not bool(event.get("revealed_group"))
                ):
                    close(pair[0], {pair[1]})
                    close(pair[1], {pair[0]})
            if event.get("revealed_group"):
                reveal_group(event["revealed_group"])
        elif event_type in {"reveal", "solution"}:
            partners: dict[str, set[str]] = {}
            for raw_pair in event.get("pairs", []):
                pair = canonical(raw_pair)
                mark(pair, "yes", event_type)
                for person, partner in (pair, pair[::-1]):
                    partners.setdefault(person, set()).add(partner)
            if event_type == "solution":
                for person, matches in partners.items():
                    close(person, matches)
            for group in event.get("complete_groups", []):
                reveal_group(group)

    for raw_pair in arrival_exclusions:
        mark(canonical(raw_pair), "no", "partners")

    raw_nights = [
        (event_week, event)
        for event_week, _, event in history
        if event["type"] == "matching_night"
    ]

    def reduce_nights() -> tuple[VillaNight, ...]:
        nights = []
        for number, (event_week, event) in enumerate(raw_nights, start=1):
            pairs = tuple(canonical(pair) for pair in event["pairs"])
            open_pairs = tuple(pair for pair in pairs if pair not in states)
            remaining = None
            if event.get("result") != "sold":
                confirmed = sum(
                    pair in states and states[pair].result == "yes" for pair in pairs
                )
                remaining = (
                    int(event["lights"])
                    - int(event.get("automatic_lights", 0))
                    - confirmed
                )
                if not 0 <= remaining <= len(open_pairs):
                    raise DataValidationError(
                        f"Villa W{week} MN {number}: {remaining} unresolved lights "
                        f"for {len(open_pairs)} open pairs"
                    )
            nights.append(VillaNight(
                number, event_week, pairs, remaining, open_pairs,
                None if event.get("result") == "sold" else int(event["lights"]),
                int(event.get("automatic_lights", 0)),
            ))
        return tuple(nights)

    while True:
        changed = False
        confirmed_partners: dict[str, set[str]] = {}
        for (a, b), state in states.items():
            if state.result == "yes":
                confirmed_partners.setdefault(a, set()).add(b)
                confirmed_partners.setdefault(b, set()).add(a)
        full = set(closed)
        for person, partners in confirmed_partners.items():
            if person in closed:
                if not partners <= closed[person]:
                    raise DataValidationError(f"Villa W{week}: extra match for {person}")
            elif len(partners) > limits[person]:
                raise DataValidationError(f"Villa W{week}: too many matches for {person}")
            elif len(partners) == limits[person]:
                full.add(person)
        for a in row_names:
            for b in bit_names:
                pair = (a, b)
                if pair not in states and (a in full or b in full):
                    changed |= mark(pair, "no", "partners")
        nights = reduce_nights()
        for night in nights:
            if night.unresolved_lights is None or not night.open_pairs:
                continue
            if night.unresolved_lights == 0:
                for pair in night.open_pairs:
                    changed |= mark(pair, "no", "matching_night")
            elif night.unresolved_lights == len(night.open_pairs):
                for pair in night.open_pairs:
                    changed |= mark(pair, "yes", "matching_night")
        if not changed:
            break

    return VillaSnapshot(week, row_names, bit_names, face_layer, states, nights)


def villa_portrait_names(data: dict[str, Any]) -> tuple[str, ...]:
    """Include newcomers so the table's cast columns stay fixed from week one."""
    names = {name for group in data["groups"].values() for name in group}
    for _, _, event in flatten_events(data):
        if event["type"] == "cast_change":
            names.update(addition["person"] for addition in event.get("add", []))
    return tuple(sorted(names, key=str.casefold))


def _confirmed_seatings(states: dict[Pair, VillaPairState]) -> dict[str, tuple[str, ...]]:
    partners: dict[str, list[str]] = {}
    for (person, partner), state in states.items():
        if state.result == "yes":
            partners.setdefault(person, []).append(partner)
    return {person: tuple(names) for person, names in partners.items()}


def build_villa_snapshot(data: dict[str, Any], week: int) -> VillaSnapshot:
    """Reconstruct a weekly table, with ten MN rows and only public knowledge."""
    all_events = flatten_events(data)
    history = [item for item in all_events if item[0] <= week]
    knowledge = _derive_villa_knowledge(data, week, history)
    row_group = data["matrix"]["rows"]
    columns = set(data["groups"][row_group])
    for _, _, event in all_events:
        if event["type"] == "cast_change":
            columns.update(
                addition["person"] for addition in event.get("add", [])
                if addition["group"] == row_group
            )
    table_columns = tuple(sorted(columns, key=str.casefold))
    table_rows = []
    night_indices = [
        index for index, (_, _, event) in enumerate(history)
        if event["type"] == "matching_night"
    ]
    for night, index in zip(knowledge.nights, night_indices):
        # Missing confirmed matches are automatic seats. Only knowledge already
        # available BEFORE this night can reconstruct them; later discoveries
        # update result markers and remaining lights, never historical seating.
        before = _derive_villa_knowledge(data, night.week, history[:index])
        seatings = {
            person: list(partners)
            for person, partners in _confirmed_seatings(before.pair_states).items()
        }
        for person, partner in night.pairs:
            partners = seatings.setdefault(person, [])
            if partner not in partners:
                partners.append(partner)
        table_rows.append(VillaTableRow(
            night.number, night.week,
            {person: tuple(partners) for person, partners in seatings.items()},
            night.lights, night.unresolved_lights, True, len(night.open_pairs),
        ))
    confirmed = _confirmed_seatings(knowledge.pair_states)
    for number in range(len(table_rows) + 1, max(10, len(table_rows)) + 1):
        table_rows.append(VillaTableRow(number, None, dict(confirmed), None, None, False))
    matchboxes = []
    for event_week, _, event in history:
        if event["type"] != "box":
            continue
        a, b = event["pair"]
        people = [a, b] if a in columns else [b, a]
        group = event.get("revealed_group")
        if group:
            # Put the shared partner in the middle of a revealed double match.
            members = list(group["members"])
            people = [members[0], group["partner"], *members[1:]]
            for person in event["pair"]:
                if person not in people:
                    people.append(person)
        matchboxes.append(VillaMatchBox(
            len(matchboxes) + 1, event_week, tuple(people), event["result"],
        ))
    return VillaSnapshot(
        knowledge.week, knowledge.row_names, knowledge.bit_names,
        knowledge.face_layer, knowledge.pair_states, knowledge.nights,
        table_columns, tuple(table_rows), tuple(matchboxes),
    )


def iter_villa_snapshots(
    data: dict[str, Any], through_week: int | None = None
) -> Iterator[VillaSnapshot]:
    """Yield exactly one end-of-week snapshot for each week with a night."""
    for week in data["weeks"]:
        number = int(week["number"])
        if through_week is not None and number > through_week:
            break
        if any(event["type"] == "matching_night" for event in week["events"]):
            yield build_villa_snapshot(data, number)

