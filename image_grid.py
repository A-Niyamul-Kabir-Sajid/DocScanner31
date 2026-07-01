"""Compose multiple OpenCV images into a single labelled grid.

This is a faithful Python-3 / NumPy port of Murtaza's Workshop
``utlis.stackImages`` helper from the classic 4-step scanner tutorial:

    * Tiles are uniformly scaled (``scale`` < 1 shrinks, > 1 enlarges).
    * Grayscale (single-channel) tiles are auto-promoted to 3-channel BGR
      so they h-stack cleanly with colour tiles.
    * When ``labels`` is supplied, a white-filled header strip is drawn at
      the top of each tile with the label text in magenta.

The function accepts either a flat list ``[img0, img1, ...]`` *or* a
2-D nested list ``[[r0c0, r0c1, ...], [r1c0, r1c1, ...]]`` and tiles
them into ``rows x cols`` blocks stacked top-to-bottom then
left-to-right.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

GridImage = Union[np.ndarray, None]
GridInput = Union[Sequence[GridImage], Sequence[Sequence[GridImage]]]


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Promote a grayscale image to 3-channel BGR (no-op for already-3ch)."""
    if img is None:
        raise ValueError("stack_images received a None tile")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def stack_images(
    img_array: GridInput,
    scale: float = 0.5,
    labels: Optional[Sequence[Sequence[str]]] = None,
) -> np.ndarray:
    """Tile ``img_array`` into one canvas.

    Parameters
    ----------
    img_array:
        Either a flat iterable of images (treated as a single row) or a
        2-D iterable of shape ``(rows, cols)``.  ``None`` entries become
        blank black tiles of the same size as the first non-None tile in
        their row.
    scale:
        Uniform resize factor applied to every tile *before* stacking.
    labels:
        Optional ``rows x cols`` matrix of short strings drawn as a
        header on each tile (top-left, magenta on white).
    """
    rows_available = bool(img_array) and isinstance(img_array[0], (list, tuple))

    if rows_available:
        rows = len(img_array)
        cols = len(img_array[0])
        # Use the first non-None tile as the size reference.
        ref_h = ref_w = 0
        for r in img_array:
            for tile in r:
                if tile is not None:
                    ref_h, ref_w = tile.shape[:2]
                    break
            if ref_h:
                break
        if ref_h == 0 or ref_w == 0:
            raise ValueError("stack_images: no valid tiles supplied")

        new_w = max(1, int(ref_w * scale))
        new_h = max(1, int(ref_h * scale))

        scaled: List[List[np.ndarray]] = []
        blank = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for r in img_array:
            row_tiles: List[np.ndarray] = []
            for tile in r:
                if tile is None:
                    row_tiles.append(blank.copy())
                    continue
                resized = cv2.resize(tile, (new_w, new_h))
                row_tiles.append(_ensure_bgr(resized))
            scaled.append(row_tiles)

        stacked_rows = [np.hstack(row) for row in scaled]
        grid = np.vstack(stacked_rows)
    else:
        # Single-row layout - resize each tile to the size of the first.
        first = next((t for t in img_array if t is not None), None)
        if first is None:
            raise ValueError("stack_images: no valid tiles supplied")
        ref_h, ref_w = first.shape[:2]
        new_w = max(1, int(ref_w * scale))
        new_h = max(1, int(ref_h * scale))

        tiles = []
        blank = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for tile in img_array:
            if tile is None:
                tiles.append(blank.copy())
                continue
            resized = cv2.resize(tile, (new_w, new_h))
            tiles.append(_ensure_bgr(resized))
        grid = np.hstack(tiles)
        rows, cols = 1, len(tiles)

    if labels:
        # White header strip per tile, with magenta text in the top-left.
        each_w = grid.shape[1] // cols
        each_h = grid.shape[0] // rows
        for r in range(rows):
            for c in range(cols):
                label = str(labels[r][c])
                x0 = c * each_w
                y0 = r * each_h
                # Background pill sized to the label width.
                cv2.rectangle(
                    grid,
                    (x0, y0),
                    (x0 + len(label) * 13 + 27, y0 + 30),
                    (255, 255, 255),
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    grid,
                    label,
                    (x0 + 10, y0 + 20),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

    return grid


__all__ = ["stack_images"]