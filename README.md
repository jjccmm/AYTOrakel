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
- `ayto_villa.py` reconstructs the public knowledge for each weekly Villa post.
- `ayto_data.json` contains cast, topology, style, and event data for every season.
- `historical_metrics.csv` contains one reproducible metrics row per season/week.
- `ayto_data_viewer.html` is a standalone viewer/editor for the season data.
- `index.html` and `web/` are the interactive static AYTO puzzle.
- `ayto_web.py` / `start_puzzle.bat` start a local puzzle preview.
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

## Villa knowledge posts

Every week containing a Matching Night also gets exactly one
`<season>/<season>_<week>_insta_Villa.png`, in 1080x1350.

For `s6vip`, `style.villa_faces` points to `insta_styles/s6vip_faces/`, containing
one transparent `<Name>.png` portrait per participant. The Villa post has ten
MN rows from week one onward and fixed columns for all eleven men (including
Laurenz). The season background remains visible; horizontal offwhite lines
join the week discs, larger faces, total-light discs, and allocation capsules.
Shapes are rendered at four times their display resolution and reduced with
Lanczos resampling for smooth edges. A solid black divider sits under the column
headers, labelled `W`, `L`, and `?`. The `L` is yellow and the `?` red, both
with a black outline.

Small week numbers are black in small offwhite discs. Total lights are yellow
with a black outline. Allocations use red text with a black outline in an
88x48 capsule with semicircular ends and a straight middle. Both use 26-point
text. Lines stop inside the allocation column. Fractions are shown without
reducing them, e.g. `3/10` becomes `2/7` after one confirmed match and two
excluded seats. Fully solved nights show `0`, avoiding `0/0`. Sold nights use
a light-blue `$` (or `$/N` for the allocation).

Confirmed matches are prefilled in future MN rows. The complete contents of
future rows, including lines, discs, week numbers, portraits, and result
markers, are composited at 50% opacity. Future light/allocation fields are
blank. Past rows retain reported seats and matches already known before that
night. Later discoveries update status markers and allocation fractions
without adding hypothetical past seats. MN rows have 74-pixel spacing, with
result markers below the portraits. Green pluses and red crosses retain the
existing AYTO colors; double matches overlap and share one result marker.

The frameless Match Box list has six compact groups per row, with only `W1`,
`W2`, etc. above each group. It includes decisions available that week, uses
overlapping portraits and one centered result symbol, and puts the shared
partner in the middle of revealed double matches. Sold boxes show a light-blue
`$` sized to match the visible height of `+` and `X`. Other seasons without
`style.villa_faces` retain the pair-grid view using their existing face layers.

`ayto_villa.py` reconstructs each week's public knowledge independently of the
exact solver: Match Boxes and reveals, partner capacity limits, and nights
with either no remaining lights or enough lights for every open pair.
Deductions propagate across all earlier nights until no further pair changes.
Automatic lights are subtracted separately from confirmed explicit seating
pairs. Complete normal matches exclude other partners; partial confirmations
and multi-matches retain any still-available partner capacity.

Each post includes every event through the end of its own week, including
events after that week's night, and never includes future information.
Participant additions and face-layer changes use that week's cast state. The
portrait-table header includes the complete cast to keep column positions fixed.
Villa posts are regenerated even when an existing solver checkpoint already
covers the week; repeated runs replace the same files. `--through-week` also
limits the Villa export. Existing probability, light, and performance posts
continue to be generated as before.

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


## Interactive AYTO puzzle

Open `start_puzzle.bat` on Windows, or run `python ayto_web.py`. The local
server opens the puzzle in your browser. It serves static files without
modifying the season JSON. Stop it with Ctrl+C.

The page fetches `ayto_data.json` and selects its last top-level entry in file
order. Portraits come from that season's `style.villa_faces` directory;
newcomers are included. No checkpoint, generated probabilities or hidden
solution is fetched. Missing portraits fall back to initials.

Tap a seated face to toggle an assumption between open and Perfect Match.
Public results and their unconditional consequences are fixed. Confirmed and
assumed matches share the Instagram green; No Matches share its red. Matched
partners appear together in the table header. Excluded faces are grayed and
locked. Every edit recomputes partner constraints and all previous nights to
a fixed point, accounting for automatic lights, sold events and allowed
double matches. An unmatched person with only one possible partner is also
resolved automatically, including pairs that never sat together. Solved
nights show an empty green allocation capsule.
Difficulty is selectable below Matching Nights, with **Medium** selected by
default. **Easy** rejects a contradictory new assumption and explains it in
a dialog. **Medium** accepts the move and highlights affected nights without
an explanation or locking other rows. **Hard** accepts it silently, hides
allocation numbers and solved-night capsules, and omits new deductions from
assumed night totals and sole remaining partners. Public facts, explicit
assumptions and partner exclusions stay visible. The complete engine still
validates the final solution in every mode. In Hard, the user must also
account for each reported night light before the solution is revealed.
Switching difficulty preserves inputs; removals and resets always remain possible.

The global Reset is the active button at the far right of the table header;
row Reset removes only assumptions entered in that exact row. The progress
bar sits immediately below the rounded header, without a visible count.

Played nights always remain in chronological week order. Shared pairs remain
global; each explicit assumption retains the row where it was entered. All
assumptions are local to this browser session and are never saved to
`ayto_data.json`. An info panel below the Instagram link explains the light
allocation fractions and the tap interaction. Matchbox photos remain colored,
including No Match results.

Both tables fit the available width without horizontal scrolling. Portraits,
numbers and symbols scale down on smaller windows and phones. The Grid is a
compact read-only view with square cells and no symbols. Impossible pairs
blend into the page background, possible pairs are blue, and Perfect Matches
are green; legend swatches have an offwhite outline. The Grid is centered
under "Perfect Match Möglichkeiten". All assumptions are entered in the
Matching Nights.

The victory dialog and confetti require a complete, consistent match graph:
every participant must be matched, all reported lights must agree, and
double-match topology must be valid. Connected match groups count once,
including three-person double matches. Confetti respects reduced-motion
preferences. Keyboard controls, visible focus, labels and live announcements
complement finger taps. The solution dialog includes the requested Buy me a
coffee button and a short thank-you message; the browser tab uses
`insta_styles/AYTO.png` as its favicon.

### GitHub Pages, when ready

The application is buildless: `index.html`, `web/`, `ayto_data.json`,
`insta_styles/AYTO.png` and the portrait directories can be served directly.
All paths are relative, including assets and the JSON request, so a repository subpath works. `.nojekyll` avoids
Jekyll processing. No npm installation or backend is needed in production.

Later, in the repository's **Settings → Pages**, select **Deploy from a
branch**, your chosen branch, and **/(root)**. Include the files above and
portrait PNGs in that branch before enabling Pages. Nothing has been published
as part of local implementation. See the official
[GitHub Pages setup guide](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)
for hosting configuration.

Checks:

```bash
node --test tests/test_web.mjs
python -m unittest discover -s tests
```
