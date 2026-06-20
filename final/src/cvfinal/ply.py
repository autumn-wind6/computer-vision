from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Vertex:
    x: float
    y: float
    z: float
    red: int
    green: int
    blue: int


def _clamp_color(value: int | float) -> int:
    return max(0, min(255, int(round(float(value)))))


def read_ascii_xyzrgb_ply(path: str | Path, default_color: Iterable[int] = (180, 180, 180)) -> list[Vertex]:
    source = Path(path)
    with open(source, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"{source} is not a PLY file")
        fmt = f.readline().strip()
        if fmt != "format ascii 1.0":
            raise ValueError(f"{source} must be ASCII PLY; got {fmt!r}")

        vertex_count: int | None = None
        in_vertex = False
        properties: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{source} has no end_header")
            stripped = line.strip()
            if stripped == "end_header":
                break
            parts = stripped.split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if in_vertex and len(parts) >= 3 and parts[0] == "property":
                properties.append(parts[-1])

        if vertex_count is None:
            raise ValueError(f"{source} has no vertex element")
        required = {"x", "y", "z"}
        if not required.issubset(properties):
            raise ValueError(f"{source} must contain x y z properties")

        default = list(default_color)
        if len(default) != 3:
            raise ValueError("default_color must have three channels")

        vertices: list[Vertex] = []
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                break
            values = line.strip().split()
            row = {name: values[i] for i, name in enumerate(properties) if i < len(values)}
            red = row.get("red", row.get("r", default[0]))
            green = row.get("green", row.get("g", default[1]))
            blue = row.get("blue", row.get("b", default[2]))
            vertices.append(
                Vertex(
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    _clamp_color(red),
                    _clamp_color(green),
                    _clamp_color(blue),
                )
            )
    return vertices


def write_ascii_xyzrgb_ply(path: str | Path, vertices: Iterable[Vertex]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(vertices)
    with open(target, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for v in rows:
            f.write(f"{v.x:.8f} {v.y:.8f} {v.z:.8f} {v.red} {v.green} {v.blue}\n")


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def euler_xyz_deg_matrix(rotation_deg: Iterable[float]) -> list[list[float]]:
    rx, ry, rz = [math.radians(float(v)) for v in rotation_deg]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    my = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    mz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    return _matmul(mz, _matmul(my, mx))


def transform_vertices(
    vertices: Iterable[Vertex],
    scale: float = 1.0,
    rotation_deg: Iterable[float] = (0.0, 0.0, 0.0),
    translation: Iterable[float] = (0.0, 0.0, 0.0),
) -> list[Vertex]:
    rot = euler_xyz_deg_matrix(rotation_deg)
    tx, ty, tz = [float(v) for v in translation]
    out: list[Vertex] = []
    for v in vertices:
        x, y, z = v.x * scale, v.y * scale, v.z * scale
        xr = rot[0][0] * x + rot[0][1] * y + rot[0][2] * z + tx
        yr = rot[1][0] * x + rot[1][1] * y + rot[1][2] * z + ty
        zr = rot[2][0] * x + rot[2][1] * y + rot[2][2] * z + tz
        out.append(Vertex(xr, yr, zr, v.red, v.green, v.blue))
    return out
