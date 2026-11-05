#!/usr/bin/env python3
"""Reconstruct a coarse world land/sea map from hand-written geographic knowledge.

This program deliberately contains no downloaded map, geographic database,
coastline package, or image-generation model.  The polygons and ellipses below
are approximate human-authored rules.  A point is land when it is inside one
of those shapes and is not inside a hand-written inland-lake hole.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

LATITUDES = list(range(89, -90, -2))
LONGITUDES = list(range(-179, 180, 2))

Point = tuple[float, float]  # longitude, latitude


# These are intentionally broad, low-resolution outlines.  They are not
# imported from GIS data: each list is a compact approximation written for the
# experiment.  Small islands are represented by ellipses below.
LAND_POLYGONS: list[list[Point]] = [
    # North America, including Mexico and a rough Central American bridge.
    [
        (-168, 65), (-150, 72), (-125, 73), (-105, 75), (-82, 72),
        (-63, 62), (-55, 52), (-61, 43), (-70, 31), (-80, 25),
        (-87, 19), (-94, 16), (-100, 21), (-108, 26), (-117, 31),
        (-122, 39), (-130, 49), (-145, 55), (-160, 58),
    ],
    # Alaska and the Aleutian-facing western edge.
    [(-171, 72), (-155, 72), (-140, 65), (-145, 57), (-160, 55), (-172, 60)],
    # Central America and the Caribbean-side isthmus.
    [(-92, 19), (-84, 18), (-78, 10), (-82, 7), (-88, 10), (-94, 15)],
    # South America.
    [
        (-81, 12), (-73, 12), (-66, 9), (-60, 5), (-52, 3), (-45, -2),
        (-38, -8), (-35, -18), (-39, -27), (-48, -34), (-55, -38),
        (-63, -46), (-70, -53), (-75, -48), (-75, -35), (-79, -20),
        (-81, -5),
    ],
    # Africa, with the Mediterranean and Red Sea margins kept approximate.
    [
        (-17, 35), (-5, 37), (10, 36), (25, 35), (35, 31), (43, 12),
        (51, 10), (49, -2), (43, -15), (35, -25), (31, -34), (19, -35),
        (8, -34), (-1, -29), (-10, -18), (-16, -4), (-17, 12),
    ],
    # Europe through Siberia and East Asia.  Peninsulas are supplemented by
    # separate shapes, while the southern boundary leaves the major seas open.
    [
        (-11, 35), (-9, 45), (0, 48), (8, 54), (20, 58), (35, 60),
        (52, 63), (75, 69), (105, 72), (135, 70), (165, 68), (180, 63),
        (180, 45), (165, 43), (150, 40), (140, 34), (130, 28),
        (121, 22), (110, 18), (101, 10), (93, 8), (86, 20), (78, 8),
        (70, 20), (60, 20), (51, 25), (42, 30), (32, 30), (25, 36),
        (15, 35), (5, 36),
    ],
    # Arabian Peninsula.
    [(35, 32), (43, 31), (55, 27), (60, 20), (52, 12), (43, 13), (38, 20)],
    # Indian subcontinent.
    [(68, 28), (78, 31), (88, 28), (93, 20), (88, 8), (80, 7), (73, 15)],
    # Mainland Southeast Asia.
    [(95, 24), (106, 23), (111, 17), (108, 8), (102, 6), (97, 12)],
    # Australia.
    [
        (113, -11), (128, -10), (142, -12), (153, -18), (154, -28),
        (149, -39), (137, -40), (124, -35), (114, -31), (112, -22),
    ],
    # Tasmania.
    [(144, -40), (149, -41), (148, -44), (143, -44), (140, -42)],
    # Antarctica: a deliberately simplified ice-covered land shelf.
    [
        (-180, -67), (-160, -70), (-140, -72), (-120, -70), (-100, -73),
        (-80, -70), (-60, -74), (-40, -70), (-20, -72), (0, -69),
        (20, -71), (40, -68), (60, -70), (80, -69), (100, -72),
        (120, -68), (140, -71), (160, -68), (180, -67), (180, -90),
        (-180, -90),
    ],
    # Greenland.
    [(-73, 60), (-61, 59), (-45, 63), (-24, 70), (-18, 78), (-28, 84),
     (-45, 84), (-59, 79), (-70, 70)],
]

# (centre longitude, centre latitude, east-west radius, north-south radius).
# These capture islands that occupy at least one 2-degree sample cell at this
# resolution; tiny islets naturally disappear between samples.
LAND_ELLIPSES: list[tuple[float, float, float, float]] = [
    (-19, 65, 4, 2),      # Iceland
    (-4, 54, 5, 6),       # Great Britain
    (-8, 53, 2, 3),       # Ireland
    (12, 42, 2, 2),       # Corsica/Sardinia area (coarse)
    (24, 39, 3, 2),       # Aegean/Cyprus group
    (138, 37, 4, 8),      # Honshu and the main Japanese arc
    (142, 45, 3, 4),      # Hokkaido
    (121, 16, 5, 7),      # Philippines
    (106, -2, 13, 7),     # Indonesia's large islands
    (117, 4, 4, 5),       # Borneo continuation
    (47, -19, 4, 9),      # Madagascar
    (174, -41, 4, 8),     # New Zealand
    (-77, 18, 4, 2),      # Hispaniola/Puerto Rico group
    (-80, 22, 4, 2),      # Cuba
    (-157, 20, 2, 2),     # Hawaii
    (167, -16, 2, 3),     # Vanuatu/Fiji-scale island group
    (179, -17, 2, 2),     # western Pacific island group
]

# Inland lakes are holes in otherwise terrestrial shapes.  They are also
# approximate and only matter where a 2-degree sample happens to hit them.
LAKE_ELLIPSES: list[tuple[float, float, float, float]] = [
    (-84, 45, 5, 2),       # Great Lakes
    (50, 42, 6, 3),        # Caspian Sea
    (60, 45, 3, 2),        # Aral Sea
    (33, -1, 3, 2),        # Lake Victoria
    (29, -6, 2, 5),        # Lake Tanganyika
    (108, 53, 3, 2),       # Lake Baikal
    (-112, 41, 3, 1),      # Great Salt Lake
    (31, 60, 3, 2),        # Lake Ladoga
    (-69, -16, 2, 1),      # Lake Titicaca
]


def point_in_polygon(lon: float, lat: float, polygon: Sequence[Point]) -> bool:
    """Return whether a point is inside a closed polygon (ray casting)."""
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = ((yi > lat) != (yj > lat))
        if crosses:
            x_at_lat = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_at_lat:
                inside = not inside
        j = i
    return inside


def point_in_ellipse(lon: float, lat: float, ellipse: tuple[float, float, float, float]) -> bool:
    cx, cy, rx, ry = ellipse
    return ((lon - cx) / rx) ** 2 + ((lat - cy) / ry) ** 2 <= 1.0


def is_land(lon: float, lat: float) -> int:
    land = any(point_in_polygon(lon, lat, p) for p in LAND_POLYGONS)
    land = land or any(point_in_ellipse(lon, lat, e) for e in LAND_ELLIPSES)
    if land and any(point_in_ellipse(lon, lat, e) for e in LAKE_ELLIPSES):
        land = False
    return int(land)


def build_matrix() -> list[list[int]]:
    return [[is_land(lon, lat) for lon in LONGITUDES] for lat in LATITUDES]


def write_outputs(matrix: list[list[int]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "world_map_matrix.csv"
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(matrix)

    # L mode is used so every pixel is exactly 0 (black) or 255 (white).
    pixels = bytes(255 if value else 0 for row in matrix for value in row)
    original = Image.frombytes("L", (len(LONGITUDES), len(LATITUDES)), pixels)
    original.save(out_dir / "world_map_180x90.png")
    enlarged = original.resize((1440, 720), resample=Image.Resampling.NEAREST)
    enlarged.save(out_dir / "world_map_1440x720.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/knowledge_world_map_20261026"),
        help="directory for the CSV and PNG deliverables",
    )
    args = parser.parse_args()
    matrix = build_matrix()
    write_outputs(matrix, args.out_dir)
    values = [value for row in matrix for value in row]
    print(f"rows={len(matrix)} cols={len(matrix[0])} values={len(values)} land={sum(values)} water={len(values)-sum(values)}")
    print(f"outputs={args.out_dir}")


if __name__ == "__main__":
    main()
