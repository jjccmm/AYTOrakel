# AYTOrakel

AYTOrakel calculates exact Perfect Match probabilities for the German RTL+
show *Are You The One?* and renders the final 1080x1350 Instagram posts.

Every possible solution is represented once as a bipartite Perfect Match graph.
A double or triple match is therefore not counted several times merely because
its members could sit in different ways during the finale.

## Project structure

- `aytorakel.py` provides the command-line interface, validation, and checkpoints.
- `ayto_solver.py` generates canonical match graphs and applies show events.
- `ayto_performance.py` builds historical metrics and the win-chance model.
- `insta_renderer.py` renders final posts directly in memory.
- `ayto_data.json` contains cast, topology, style, and event data for every season.
- `historical_metrics.csv` contains one reproducible metrics row per season/week.
- `ayto_data_viewer.html` is a standalone viewer/editor for the season data.
- `insta_styles/` contains backgrounds, fonts, and cast-specific face layers.
- `tests/` contains the standard-library `unittest` suite.

## Installation and use

```bash
pip install -r requirements.txt
python aytorakel.py s6vip
```

Useful commands:

```bash
# Ignore a compatible checkpoint and calculate the season again
python aytorakel.py s4vip --from-scratch

# Stop after a selected week
python aytorakel.py s4vip --through-week 3

# Validate data and artwork without calculating permutations
python aytorakel.py s4vip --validate

# Regenerate every season
python aytorakel.py --all --from-scratch

# Rebuild historical metrics without loading or rendering artwork
python aytorakel.py --build-history --from-scratch

# Run the tests
python -m unittest discover -s tests -v
```

### Season data viewer

On Windows, double-click `start_data_viewer.bat`; alternatively run
`python ayto_data_viewer.py`. The viewer opens in the browser, loads
`ayto_data.json` automatically, and saves changes atomically back to that file.
It uses only Python's standard library and needs no npm, packages, or build
step. It can add, duplicate, rename, and delete seasons; edit all season
settings; and add, copy, reorder, or delete weeks and events. Match Boxes and
Matching Nights have forms with participant and result/light drop-downs.
Special cases remain available through a collapsed JSON editor, while the raw
season and complete-file views keep arbitrary future fields editable.

The HTML file can still be opened directly as a fallback; browser security then
requires selecting the JSON via **Open JSON**, and **Save** may download a new
copy instead of overwriting it. `Ctrl+S` uses the same save action. The editor
performs quick structural checks; run
`python aytorakel.py <season> --validate` for the full solver and artwork
validation before calculating a season.

The newest successful checkpoint is stored inside the season directory and is
ignored by Git. A checkpoint is accepted only if its saved events are an exact
prefix of the current data. Appending events resumes from it; editing history
starts a fresh calculation.

Final PNGs are written to `<season>/insta/`. Matplotlib helper plots and CSV
intermediates are never written. If a configured face layer is missing,
validation creates a labelled 1080x1350 authoring guide in
`insta_styles/face_layer_templates/` and exits before the expensive calculation.

The migrated `s6vip` data currently expects two new transparent overlays:
`ayto_s6vip_initial.png` for the original 10/10 cast and
`ayto_s6vip_laurenz.png` for the later cast state. Their generated guides are in
`insta_styles/face_layer_templates/`; place finished overlays in
`insta_styles/image_face_layers/` and rerun validation.

## Event data

Matching Nights use explicit pairs:

```json
{
  "type": "matching_night",
  "pairs": [["Person A", "Person B"]],
  "lights": 1,
  "automatic_lights": 0
}
```

A sold Matching Night uses `result: "sold"`, keeps its explicit seating pairs,
and omits `lights`. It neither filters solutions nor creates a light post.

Match Boxes use `result: "yes"`, `"no"`, or `"sold"`. A normal `yes` is a
complete one-to-one Perfect Match by default. Set `complete_pair: false` only
when the pair may still belong to a multi-match; a `revealed_group` implies that
automatically. Cast changes may use
`match_update.mode: "rebuild"` when an arrival's role is unknown, or `"extend"`
when the arrival is guaranteed to add an edge to an existing graph. See
`ayto_data.json` for complete season examples.

All participant images belong to RTL. Follow
[@AYTOrakel](https://www.instagram.com/AYTOrakel) for the published results.

## Historical performance

Every ongoing-season week gets an additional `*_insta_Performance.png`. It
compares visible results and solver-only information metrics with completed
seasons. The historical win chance uses four transparent comparisons at the
same Matching Night: lights that night, cumulative lights, confirmed Match
Boxes, and puzzle progress. For each metric it reports how many same-or-worse
past seasons still won; the displayed chance is their mean and the uncertainty
is their standard deviation. No machine-learning dependency is required.

After changing any season or event, rebuild `historical_metrics.csv` with
`python aytorakel.py --build-history --from-scratch`. Normal runs for an ongoing
season reject missing or stale historical data instead of mixing incompatible
calculations. The history command never checks face layers or creates posts.
