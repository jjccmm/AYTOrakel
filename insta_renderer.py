"""Instagram image rendering for AYTOrakel.

All charts are rendered in memory.  Only finished 1080x1350 post images (and
explicit face-layer authoring guides) are written to disk.
"""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ayto_solver import NightRecord, Snapshot, flatten_events
from ayto_performance import PerformanceRecord, WinPrediction


POST_SIZE = (1080, 1350)
FONT_PATH = Path("insta_styles/DelaGothicOne-Regular.ttf")
ROCKET = LinearSegmentedColormap.from_list(
    "ayto_rocket",
    [
        "#03051a",
        "#30173a",
        "#611f53",
        "#971c5b",
        "#cb1b4f",
        "#ec4c3e",
        "#f58860",
        "#f6bc99",
        "#faebdd",
    ],
)
PROBABILITY_FONT_SIZE = 14
SUMMARY_FONT_SIZE = 13


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def _season_label(season: str) -> str:
    number = season.replace("vip", "").replace("s", "")
    suffix = " VIP" if "vip" in season else ""
    return f"AYTO S{number}{suffix}"


def _background(style: dict[str, Any]) -> Image.Image:
    with Image.open(style["background"]) as source:
        image = source.convert("RGB")
    if image.size != POST_SIZE:
        raise ValueError(f"Background must be {POST_SIZE}, got {image.size}")
    return image


def _safe_filename_part(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "-", value).strip().rstrip(".")


def _save_png(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.save(temporary, format="PNG")
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1 * (attempt + 1))
    return path


def heatmap_bounds(column_count: int, row_count: int) -> tuple[int, int, int, int]:
    cell = min(810 / column_count, 891 / row_count)
    return 220, 270, round(column_count * cell), round(row_count * cell)


def _figure_image(
    fig: Figure, width: int, height: int, *, transparent: bool = True
) -> Image.Image:
    stream = BytesIO()
    fig.set_size_inches(width / 100, height / 100)
    fig.savefig(
        stream,
        format="png",
        dpi=100,
        transparent=transparent,
        bbox_inches=None,
        pad_inches=0,
    )
    stream.seek(0)
    return Image.open(stream).convert("RGBA")


def _annotation_color(value: float) -> str:
    """Use Seaborn's former luminance rule for readable heatmap labels."""
    rgb = np.asarray(ROCKET(value / 100.0)[:3])
    linear = np.where(
        rgb <= 0.03928,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )
    luminance = float(np.dot(linear, [0.2126, 0.7152, 0.0722]))
    return "#262626" if luminance > 0.408 else "white"


def _probability_label(value: float) -> str:
    if value == 0:
        return ""
    if value < 1:
        label = f"{value:.3f}"
        if label == "0.000":
            return "<0.001"
        return label.rstrip("0").rstrip(".")
    return f"{value:.0f}"


def _probability_heatmap(probabilities: np.ndarray, width: int, height: int) -> Image.Image:
    values = probabilities.T
    fig = Figure(frameon=False)
    axis = fig.add_axes((0, 0, 1, 1))
    axis.imshow(values, cmap=ROCKET, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    axis.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1)
    axis.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if value == 0:
                continue
            axis.text(
                x,
                y,
                _probability_label(float(value)),
                ha="center",
                va="center",
                color=_annotation_color(float(value)),
                fontsize=PROBABILITY_FONT_SIZE,
            )
    return _figure_image(fig, width, height)


def _pair_position(snapshot: Snapshot, pair: list[str]) -> tuple[int, int] | None:
    if pair[0] in snapshot.row_names and pair[1] in snapshot.bit_names:
        return snapshot.row_names.index(pair[0]), snapshot.bit_names.index(pair[1])
    if pair[1] in snapshot.row_names and pair[0] in snapshot.bit_names:
        return snapshot.row_names.index(pair[1]), snapshot.bit_names.index(pair[0])
    return None


def _draw_event_markers(
    image: Image.Image, snapshot: Snapshot, season_data: dict[str, Any]
) -> None:
    x0, y0, width, height = heatmap_bounds(len(snapshot.row_names), len(snapshot.bit_names))
    cell_w = width / len(snapshot.row_names)
    cell_h = height / len(snapshot.bit_names)
    draw = ImageDraw.Draw(image)
    events = flatten_events(season_data)
    history = events[: snapshot.event_index + 1] if snapshot.event_index >= 0 else []
    night_counts: dict[tuple[int, int], int] = {}
    last_night: list[list[str]] = []
    for _, _, event in history:
        if event["type"] == "matching_night":
            last_night = event["pairs"]
            for pair in event["pairs"]:
                position = _pair_position(snapshot, pair)
                if position:
                    night_counts[position] = night_counts.get(position, 0) + 1
    for (x, y), count in night_counts.items():
        radius = max(2, round(min(cell_w, cell_h) * 0.035))
        point_width = radius * 2
        first_center = x0 + x * cell_w + point_width + 2
        last_center = x0 + (x + 1) * cell_w - point_width
        step = (last_center - first_center) / 9
        for dot in range(min(count, 10)):
            cx = first_center + dot * step
            cy = y0 + (y + 0.1) * cell_h
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill="yellow",
                outline="black",
                width=1,
            )
    if snapshot.event_type == "matching_night":
        for pair in last_night:
            position = _pair_position(snapshot, pair)
            if position:
                x, y = position
                draw.rectangle(
                    (
                        round(x0 + x * cell_w),
                        round(y0 + y * cell_h),
                        round(x0 + (x + 1) * cell_w),
                        round(y0 + (y + 1) * cell_h),
                    ),
                    outline="yellow",
                    width=4,
                )
    for _, _, event in history:
        if event["type"] != "box":
            continue
        position = _pair_position(snapshot, event["pair"])
        if not position:
            continue
        x, y = position
        cx = x0 + (x + 0.78) * cell_w
        cy = y0 + (y + 0.78) * cell_h
        result = event["result"]
        marker = {"yes": "+", "no": "×", "sold": "$"}[result]
        color = {"yes": "green", "no": "red", "sold": "#66ccff"}[result]
        if result == "no":
            cx = x0 + (x + 0.5) * cell_w
            cy = y0 + (y + 0.5) * cell_h - 5
            size = max(28, round(min(cell_w, cell_h) * 0.82))
        else:
            scale = 0.30 if result == "sold" else 0.45
            size = max(16, round(min(cell_w, cell_h) * scale))
        draw.text(
            (cx, cy),
            marker,
            font=_font(size),
            fill=color,
            stroke_width=2,
            stroke_fill="black",
            anchor="mm",
        )


def _draw_header(
    image: Image.Image, season: str, week: int, title: str, subtitle: str | None = None
) -> None:
    draw = ImageDraw.Draw(image)
    title_size = 85
    while title_size > 42:
        title_font = _font(title_size)
        bounds = draw.textbbox((0, 0), title, font=title_font, stroke_width=7)
        if bounds[2] - bounds[0] <= image.width - 100:
            break
        title_size -= 2
    draw.text((image.width / 2, 60), f"{_season_label(season)} W{week}", fill="white", font=_font(85), stroke_width=7, stroke_fill="black", anchor="mm")
    draw.text((image.width / 2, 155), title, fill="red", font=title_font, stroke_width=7, stroke_fill="black", anchor="mm")
    if subtitle:
        draw.text((image.width / 2, 240), subtitle, fill="white", font=_font(30), stroke_width=5, stroke_fill="black", anchor="mm")
    draw.text((10, image.height - 30), "@AYTOrakel", fill="white", font=_font(18), stroke_width=2, stroke_fill="black", anchor="la")


def render_cover(season: str, week: int, style: dict[str, Any], output_dir: Path) -> Path:
    image = _background(style)
    draw = ImageDraw.Draw(image)
    draw.text((image.width / 2, image.height / 2 - 120), "AYTO", fill="white", font=_font(90), stroke_width=7, stroke_fill="black", anchor="mm")
    draw.text((image.width / 2, image.height / 2), _season_label(season).replace("AYTO ", ""), fill="white", font=_font(90), stroke_width=7, stroke_fill="black", anchor="mm")
    draw.text((image.width / 2, image.height / 2 + 120), f"W{week}", fill="red", font=_font(90), stroke_width=7, stroke_fill="black", anchor="mm")
    draw.text((10, image.height - 30), "@AYTOrakel", fill="white", font=_font(18), stroke_width=2, stroke_fill="black", anchor="la")
    path = output_dir / f"{season}_{week}_0_insta_cover.png"
    return _save_png(image, path)


def render_probability_post(
    season: str,
    season_data: dict[str, Any],
    snapshot: Snapshot,
    output_dir: Path,
) -> Path:
    image = _background(season_data["style"])
    subtitle = f"Mögliche Kombinationen: {snapshot.solution_count:,}".replace(",", ".")
    _draw_header(image, season, snapshot.week, snapshot.title, subtitle)
    bounds = heatmap_bounds(len(snapshot.row_names), len(snapshot.bit_names))
    x, y, width, height = bounds
    heatmap = _probability_heatmap(snapshot.probabilities, width, height)
    border = Image.new("RGB", (width + 14, height + 14), "white")
    border.paste(heatmap.convert("RGB"), (7, 7))
    image.paste(border, (x - 7, y - 7))
    _draw_event_markers(image, snapshot, season_data)
    with Image.open(snapshot.face_layer) as source:
        face = source.convert("RGBA")
    image.paste(face, (0, 0), face)
    suffix = _safe_filename_part(snapshot.title)
    path = output_dir / f"{season}_{snapshot.week}_{snapshot.event_number}_insta_{suffix}.png"
    return _save_png(image, path)


def render_light_post(
    season: str, season_data: dict[str, Any], snapshot: Snapshot, output_dir: Path
) -> Path:
    if snapshot.light_distribution is None:
        raise ValueError("A light post requires a light distribution")
    image = _background(season_data["style"])
    _draw_header(image, season, snapshot.week, "Matching Night")
    draw = ImageDraw.Draw(image)
    draw.text(
        (image.width / 2, 260),
        "Wahrscheinlichkeiten für Lichter",
        fill="white",
        font=_font(50),
        stroke_width=7,
        stroke_fill="black",
        anchor="mm",
    )
    probabilities = snapshot.light_distribution
    plot_left, plot_right = 125, 1030
    plot_top, plot_bottom = 387, 1092

    for percentage in range(0, 101, 20):
        y = plot_bottom - (plot_bottom - plot_top) * percentage / 100
        draw.text(
            (65, y),
            f"{percentage}%",
            fill="white",
            font=_font(30),
            stroke_width=7,
            stroke_fill="black",
            anchor="mm",
        )
        draw.line((plot_left, y, plot_right, y), fill="white", width=2)

    cell_width = (plot_right - plot_left) / len(probabilities)
    bar_width = cell_width * 0.68
    for light, probability in enumerate(probabilities):
        center = plot_left + (light + 0.5) * cell_width
        top = plot_bottom - float(probability) * (plot_bottom - plot_top)
        draw.rectangle(
            (center - bar_width / 2, top, center + bar_width / 2, plot_bottom),
            fill="yellow",
            outline="#222222",
            width=2,
        )

    confirmed_matches = sum(
        event["type"] == "box" and event.get("result") == "yes"
        for _, _, event in flatten_events(season_data)[: snapshot.event_index + 1]
    )
    for light in range(len(probabilities)):
        color = "yellow" if light == snapshot.actual_lights else "white"
        center = plot_left + (light + 0.5) * cell_width
        draw.text(
            (center, 1120),
            str(light),
            fill=color,
            font=_font(50),
            stroke_width=7,
            stroke_fill="black",
            anchor="mm",
        )
        if light == confirmed_matches:
            draw.text(
                (center, 1170),
                "Black",
                fill="white",
                font=_font(18),
                stroke_width=3,
                stroke_fill="black",
                anchor="mm",
            )
            draw.text(
                (center, 1192),
                "Out",
                fill="white",
                font=_font(18),
                stroke_width=3,
                stroke_fill="black",
                anchor="mm",
            )
    path = output_dir / f"{season}_{snapshot.week}_{snapshot.event_number}_insta_lights.png"
    return _save_png(image, path)


def render_combinations_post(
    season: str,
    season_data: dict[str, Any],
    snapshot: Snapshot,
    output_dir: Path,
) -> Path | None:
    sample = snapshot.combination_sample
    if sample is None or len(sample) == 0:
        return None
    image = _background(season_data["style"])
    _draw_header(image, season, snapshot.week, snapshot.title, "Verbleibende Kombinationen")
    draw = ImageDraw.Draw(image)
    count = len(snapshot.row_names)
    cell_width = 950 / count
    for column, name in enumerate(snapshot.row_names):
        draw.text((65 + (column + 0.5) * cell_width, 290), name, fill="red", font=_font(14), stroke_width=2, stroke_fill="black", anchor="mm")
    for row, solution in enumerate(sample):
        for column, mask in enumerate(solution):
            names = [name for bit, name in enumerate(snapshot.bit_names) if int(mask) & (1 << bit)]
            draw.text((65 + (column + 0.5) * cell_width, 330 + row * 39), "/".join(names), fill="white", font=_font(10 if len(names) > 1 else 14), stroke_width=2, stroke_fill="black", anchor="mm")
    path = output_dir / f"{season}_{snapshot.week}_{snapshot.event_number}_insta_remaining.png"
    return _save_png(image, path)


def render_summary(
    season: str,
    season_data: dict[str, Any],
    records: list[NightRecord],
    output_dir: Path,
    last_week: int,
) -> Path | None:
    if not records or last_week < 10:
        return None
    max_lights = max(
        10,
        max(max(len(record.distribution) - 1, record.actual_lights) for record in records),
    )
    column_count = 10
    matrix = np.zeros((max_lights + 1, column_count), dtype=np.float64)
    for column, record in enumerate(records):
        if column >= column_count:
            break
        matrix[: len(record.distribution), column] = record.distribution * 100

    fig = Figure(frameon=False)
    axis = fig.add_axes((0, 0, 1, 1))
    shown = matrix[::-1]
    axis.imshow(shown, cmap=ROCKET, vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    axis.set_xticks(np.arange(-0.5, shown.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, shown.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1)
    axis.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)
    for y in range(shown.shape[0]):
        for x in range(shown.shape[1]):
            value = shown[y, x]
            if value:
                axis.text(
                    x,
                    y,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    color=_annotation_color(float(value)),
                    fontsize=SUMMARY_FONT_SIZE,
                )

    expected_x: list[int] = []
    expected_y: list[float] = []
    for column, record in enumerate(records[:column_count]):
        lights = np.arange(len(record.distribution))
        expected_x.append(column)
        expected_y.append(max_lights - float(np.sum(lights * record.distribution)))
        y = max_lights - record.actual_lights
        axis.plot(
            [column - 0.5, column + 0.5, column + 0.5, column - 0.5, column - 0.5],
            [y - 0.5, y - 0.5, y + 0.5, y + 0.5, y - 0.5],
            color="yellow",
            linewidth=3,
        )
    axis.plot(expected_x, expected_y, color="yellow", linewidth=2)

    perfect_match_weeks = [
        week
        for week, _, event in flatten_events(season_data)
        if week <= 10 and event["type"] == "box" and event.get("result") == "yes"
    ]
    for index, week in enumerate(perfect_match_weeks):
        columns = np.arange(max(0, week - 1), column_count)
        axis.scatter(
            columns,
            np.full(len(columns), max_lights - index),
            marker="P",
            s=120,
            facecolor="white",
            edgecolor="green",
            linewidth=2,
        )

    chart = _figure_image(fig, 810, 891)
    image = _background(season_data["style"])
    _draw_header(image, season, 10, "Zusammenfassung")
    image.paste(chart, (150, 320), chart)
    draw = ImageDraw.Draw(image)
    draw.text(
        (image.width / 2, 260),
        "Wahrscheinlichkeiten für Lichter",
        fill="white",
        font=_font(50),
        stroke_width=7,
        stroke_fill="black",
        anchor="mm",
    )
    cell_width = 810 / column_count
    cell_height = 891 / (max_lights + 1)
    for column in range(column_count):
        draw.text(
            (150 + (column + 0.5) * cell_width, 1230),
            str(column + 1),
            fill="white",
            font=_font(30),
            stroke_width=7,
            stroke_fill="black",
            anchor="mm",
        )
    for light in range(max_lights + 1):
        draw.text(
            (115, 320 + (max_lights - light + 0.5) * cell_height),
            str(light),
            fill="yellow",
            font=_font(30),
            stroke_width=7,
            stroke_fill="black",
            anchor="mm",
        )
    draw.text(
        (540, 1280),
        "Woche",
        fill="white",
        font=_font(50),
        stroke_width=7,
        stroke_fill="black",
        anchor="mm",
    )
    draw.text(
        (78, 330),
        "Lichter",
        fill="yellow",
        font=_font(30),
        stroke_width=2,
        stroke_fill="black",
        anchor="mm",
    )
    path = output_dir / f"{season}_11_0_insta_summary.png"
    return _save_png(image, path)


def render_performance_post(
    season: str,
    season_data: dict[str, Any],
    current: PerformanceRecord,
    history: list[PerformanceRecord],
    prediction: WinPrediction,
    output_dir: Path,
) -> Path:
    """Render observable and solver-only historical comparisons."""
    image = _background(season_data["style"])
    _draw_header(image, season, current.week, "Performance")
    draw = ImageDraw.Draw(image)
    by_season: dict[str, list[PerformanceRecord]] = {}
    for record in history:
        by_season.setdefault(record.season, []).append(record)
    for records in by_season.values():
        records.sort(key=lambda record: record.week)
    current_records = [
        record for record in by_season.get(season, []) if record.week <= current.week
    ]
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7B61A8", "#555555")

    def family(value: str) -> str:
        return value.replace("vip", "")

    def season_color(value: str) -> str:
        match = re.search(r"\d+", family(value))
        index = int(match.group()) - 1 if match else 0
        return palette[index % len(palette)]

    def line_style(value: str) -> str:
        return "--" if "vip" in value.lower() else "-"

    def style_axis(axis, percent: bool = False, maximum: int | None = None) -> None:
        axis.set_facecolor("white")
        axis.set_xlim(1, 10)
        axis.set_xticks(range(1, 11))
        if percent:
            axis.set_ylim(0, 105)
            axis.set_yticks([0, 50, 100])
            axis.set_yticklabels(["0%", "50%", "100%"])
        elif maximum is not None:
            axis.set_ylim(0, maximum)
            axis.set_yticks(range(0, maximum + 1, max(1, maximum // 5)))
        axis.grid(axis="y", color="#b8b8b8", alpha=0.65, linewidth=0.8)
        axis.tick_params(colors="black", labelsize=8, length=0)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.1)

    def historical_rows(records: list[PerformanceRecord]) -> list[PerformanceRecord]:
        if not records or records[0].status != "completed":
            return []
        end = records[0].competitive_end_week or 10
        return [record for record in records if record.week <= end]

    visible = Figure(frameon=True, facecolor="white")
    box_axis = visible.add_axes((0.07, 0.18, 0.40, 0.70))
    light_axis = visible.add_axes((0.57, 0.18, 0.40, 0.70))
    max_boxes = max(5, max((record.confirmed_boxes for record in history), default=0))
    for other, records in sorted(by_season.items()):
        shown = historical_rows(records)
        if other == season or not shown:
            continue
        box_axis.plot([r.week for r in shown], [r.confirmed_boxes for r in shown],
                      color=season_color(other), linestyle=line_style(other),
                      alpha=0.80, linewidth=1.15)
        double = next((r for r in shown if r.double_match_found), None)
        if double:
            box_axis.scatter([double.week], [double.confirmed_boxes], marker="*", s=25,
                             color=season_color(other), alpha=0.80)
        nights = [r for r in shown if r.week_lights is not None]
        light_axis.plot([r.week for r in nights], [r.week_lights for r in nights],
                        color=season_color(other), linestyle=line_style(other),
                        alpha=0.80, linewidth=1.15)
    box_axis.plot([r.week for r in current_records], [r.confirmed_boxes for r in current_records],
                  color="#E60000", linestyle="-", linewidth=4,
                  marker="o", markersize=4.5, markeredgecolor="black", zorder=20)
    current_double = next((r for r in current_records if r.double_match_found), None)
    if current_double:
        box_axis.scatter([current_double.week], [current_double.confirmed_boxes], marker="*",
                         s=115, color="#FFE900", edgecolor="black", linewidth=1, zorder=30)
    current_nights = [r for r in current_records if r.week_lights is not None]
    light_axis.plot([r.week for r in current_nights], [r.week_lights for r in current_nights],
                    color="#E60000", linestyle="-", linewidth=4,
                    marker="o", markersize=4.5, markeredgecolor="black", zorder=20)
    style_axis(box_axis, maximum=max_boxes)
    style_axis(light_axis, maximum=current.target_lights)
    box_axis.set_title("Bestätigte Match Boxes", color="black", fontsize=12, pad=5)
    light_axis.set_title("Lichter pro Nacht", color="black", fontsize=12, pad=5)
    box_axis.text(0.02, -0.21, "★ Double Match", transform=box_axis.transAxes,
                  color="black", fontsize=7)
    chart = _figure_image(visible, 920, 265, transparent=False)
    image.paste(chart, (80, 240), chart)

    hidden = Figure(frameon=True, facecolor="white")
    progress_axis = hidden.add_axes((0.07, 0.54, 0.90, 0.42))
    box_info_axis = hidden.add_axes((0.07, 0.06, 0.40, 0.32))
    night_info_axis = hidden.add_axes((0.57, 0.06, 0.40, 0.32))
    for other, records in sorted(by_season.items()):
        shown = historical_rows(records)
        if other == season or not shown:
            continue
        for axis, getter in (
            (progress_axis, lambda r: r.information_progress),
            (box_info_axis, lambda r: r.box_information_gain),
            (night_info_axis, lambda r: r.night_information_gain),
        ):
            axis.plot([r.week for r in shown], [getter(r) * 100 for r in shown],
                      color=season_color(other), linestyle=line_style(other),
                      alpha=0.80, linewidth=1.05)
    for axis, getter in (
        (progress_axis, lambda r: r.information_progress),
        (box_info_axis, lambda r: r.box_information_gain),
        (night_info_axis, lambda r: r.night_information_gain),
    ):
        axis.plot([r.week for r in current_records], [getter(r) * 100 for r in current_records],
                  color="#E60000", linestyle="-", linewidth=4,
                  marker="o", markersize=4.5, markeredgecolor="black", zorder=20)
        style_axis(axis, percent=True)
    progress_axis.set_title("AYTOrakel Lösung", color="black", fontsize=12, pad=4)
    box_info_axis.set_title("Info durch Match Boxes", color="black", fontsize=11, pad=4)
    night_info_axis.set_title("Info durch Matching Nights", color="black", fontsize=11, pad=4)
    box_info_axis.tick_params(labelleft=False)
    night_info_axis.tick_params(labelleft=False)

    legend_handles = [
        Line2D([0], [0], color=season_color(value), linewidth=1.8, label=value.upper())
        for value in sorted(
            {family(value) for value in by_season},
            key=lambda value: int(re.search(r"\d+", value).group()) if re.search(r"\d+", value) else 0,
        )
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color="black", linewidth=1.4, linestyle="-", label="Normal"),
            Line2D([0], [0], color="black", linewidth=1.4, linestyle="--", label="VIP"),
            Line2D([0], [0], color="#E60000", linewidth=4, linestyle="-", label="Aktuell"),
        ]
    )
    progress_axis.legend(
        handles=legend_handles,
        loc="lower right",
        ncol=3,
        fontsize=5.8,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.92,
        handlelength=2.3,
        columnspacing=0.9,
    )
    chart = _figure_image(hidden, 920, 465, transparent=False)
    image.paste(chart, (80, 505), chart)

    draw = ImageDraw.Draw(image)

    chance = round(prediction.probability * 100)
    deviation = round(prediction.standard_deviation * 100)
    draw.text((540, 1082), "HISTORISCHE GEWINNCHANCE", fill="white", font=_font(21),
              stroke_width=3, stroke_fill="black", anchor="mm")
    draw.text((540, 1130), f"{chance} %  ± {deviation}", fill="yellow", font=_font(43),
              stroke_width=5, stroke_fill="black", anchor="mm")
    gauge_left, gauge_right, gauge_y = 150, 930, 1190
    draw.line((gauge_left, gauge_y, gauge_right, gauge_y), fill="#dddddd", width=7)
    bounded_low = max(0.0, min(1.0, prediction.low))
    bounded_high = max(bounded_low, min(1.0, prediction.high))
    band_left = gauge_left + (gauge_right - gauge_left) * bounded_low
    band_right = gauge_left + (gauge_right - gauge_left) * bounded_high
    draw.rounded_rectangle((band_left, gauge_y - 10, band_right, gauge_y + 10),
                           radius=10, fill="#999999", outline="white", width=1)
    bounded_probability = max(0.0, min(1.0, prediction.probability))
    marker_x = gauge_left + (gauge_right - gauge_left) * bounded_probability
    draw.ellipse((marker_x - 12, gauge_y - 12, marker_x + 12, gauge_y + 12),
                 fill="yellow", outline="black", width=2)
    draw.text((gauge_left - 15, gauge_y), "0%", fill="white", font=_font(13),
              stroke_width=2, stroke_fill="black", anchor="rm")
    draw.text((gauge_right + 15, gauge_y), "100%", fill="white", font=_font(13),
              stroke_width=2, stroke_fill="black", anchor="lm")
    draw.text((540, 1230), "Gewonnen bei gleichem oder schlechterem Stand", fill="white",
              font=_font(14), stroke_width=2, stroke_fill="black", anchor="mm")
    week_data = next(week for week in season_data["weeks"] if int(week["number"]) == current.week)
    event_number = len(week_data["events"]) + 1
    path = output_dir / f"{season}_{current.week}_{event_number}_insta_Performance.png"
    return _save_png(image, path)


def render_face_layer_guide(
    season: str,
    style: dict[str, Any],
    expected_layer: Path,
    row_names: Iterable[str],
    bit_names: Iterable[str],
) -> Path:
    rows = list(row_names)
    bits = list(bit_names)
    image = _background(style)
    _draw_header(image, season, 0, "Face Layer Guide", expected_layer.name)
    x, y, width, height = heatmap_bounds(len(rows), len(bits))
    draw = ImageDraw.Draw(image)
    draw.rectangle((x, y, x + width, y + height), fill="#25152e", outline="yellow", width=5)
    cell_w = width / len(rows)
    cell_h = height / len(bits)
    for index, name in enumerate(rows):
        draw.line((x + index * cell_w, y, x + index * cell_w, y + height), fill="white", width=1)
        draw.text((x + (index + 0.5) * cell_w, y + height + 35), name, fill="white", font=_font(13), anchor="mm")
    for index, name in enumerate(bits):
        draw.line((x, y + index * cell_h, x + width, y + index * cell_h), fill="white", width=1)
        draw.text((x - 12, y + (index + 0.5) * cell_h), name, fill="white", font=_font(13), anchor="rm")
    guide_dir = Path("insta_styles/face_layer_templates")
    guide_dir.mkdir(parents=True, exist_ok=True)
    path = guide_dir / f"{expected_layer.stem}_guide.png"
    return _save_png(image, path)
