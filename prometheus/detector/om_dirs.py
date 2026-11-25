"""Directions that PPC averages module responses over."""

from pathlib import Path
from typing import Union

import numpy as np

#: Subdivisions of the icosahedron; 3 gives the 642 directions PPC usually reads.
_DEFAULT_LEVEL = 3


def sphere_directions(level: int = _DEFAULT_LEVEL) -> np.ndarray:
    """Return roughly uniform unit vectors from a subdivided icosahedron.

    Parameters
    ----------
    level : int, optional
        Number of times each triangle is split into four.

    Returns
    -------
    np.ndarray
        ``10 * 4**level + 2`` unit vectors, sorted by component.
    """
    golden = (1 + np.sqrt(5)) / 2
    vertices = []
    for i in range(3):
        for a in (-golden, golden):
            for b in (-1.0, 1.0):
                vertex = np.zeros(3)
                vertex[i] = a
                vertex[(i + 1) % 3] = b
                vertices.append(vertex / np.hypot(1, golden))
    vertices = np.array(vertices)

    # Neighboring vertices are the only pairs closer than 90 degrees.
    near = vertices @ vertices.T > 0
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if near[i, j] and near[j, k] and near[i, k]:
                    faces.append(vertices[[i, j, k]])
    triangles = np.array(faces)

    for _ in range(level):
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        middles = []
        for p, q in ((a, b), (b, c), (c, a)):
            middle = p + q
            middles.append(middle / np.linalg.norm(middle, axis=1, keepdims=True))
        ab, bc, ca = middles
        triangles = np.concatenate(
            [np.stack(t, axis=1) for t in ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))]
        )
    # Adding 0 turns -0 into 0, so that equal points compare equal.
    return np.unique(np.round(triangles.reshape(-1, 3), 9) + 0.0, axis=0)


def write_om_dirs(path: Union[str, Path], level: int = _DEFAULT_LEVEL) -> None:
    """Write an ``om.dirs`` file: index and unit vector on each line."""
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(sphere_directions(level)):
            f.write(f"{i} {x:.9g} {y:.9g} {z:.9g}\n")
