#!/usr/bin/env python3
"""Convert final geometry from Quantum ESPRESSO pw.x output to CIF (P 1)."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

BOHR_TO_ANG = 0.529177210903


class QEParseError(RuntimeError):
    pass


Vector = tuple[float, float, float]
Matrix3 = tuple[Vector, Vector, Vector]


def v_add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_scale(v: Vector, s: float) -> Vector:
    return (v[0] * s, v[1] * s, v[2] * s)


def v_dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_norm(v: Vector) -> float:
    return math.sqrt(v_dot(v, v))


def row_vec_times_matrix(v: Vector, m: Matrix3) -> Vector:
    return (
        v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
        v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
        v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
    )


def matrix_inverse_3x3(m: Matrix3) -> Matrix3:
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(det) < 1e-16:
        raise QEParseError("Cell matrix is singular; cannot convert to fractional coordinates.")

    inv = (
        ((e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det),
        ((f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det),
        ((d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det),
    )
    return inv


def _parse_alat_angstrom(lines: list[str]) -> float | None:
    pattern = re.compile(r"lattice parameter \(alat\)\s*=\s*([0-9EeDd+\-.]+)\s*a\.u\.")
    for line in lines:
        m = pattern.search(line)
        if m:
            return float(m.group(1).replace("D", "E").replace("d", "e")) * BOHR_TO_ANG
    return None


def _parse_nat(lines: list[str]) -> int | None:
    pattern = re.compile(r"number of atoms/cell\s*=\s*(\d+)")
    for line in lines:
        m = pattern.search(line)
        if m:
            return int(m.group(1))
    return None


def _last_block_index(lines: list[str], prefix: str) -> int:
    last = -1
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(prefix):
            last = i
    if last == -1:
        raise QEParseError(f"No '{prefix.upper()}' block found.")
    return last


def _extract_header_unit(header_line: str) -> str:
    m = re.search(r"\(([^)]*)\)", header_line)
    if not m:
        return ""
    return m.group(1).strip().lower()


def _parse_vec(line: str) -> Vector:
    parts = line.split()
    if len(parts) < 3:
        raise QEParseError("Malformed vector line.")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _extract_cell(lines: list[str], start_idx: int, alat_ang: float | None) -> Matrix3:
    header = lines[start_idx]
    unit_raw = _extract_header_unit(header)
    try:
        rows = (
            _parse_vec(lines[start_idx + 1]),
            _parse_vec(lines[start_idx + 2]),
            _parse_vec(lines[start_idx + 3]),
        )
    except IndexError as exc:
        raise QEParseError("Incomplete CELL_PARAMETERS block.") from exc

    if unit_raw.startswith("angstrom"):
        return rows
    if unit_raw.startswith("bohr"):
        return tuple(v_scale(r, BOHR_TO_ANG) for r in rows)  # type: ignore[return-value]

    if unit_raw.startswith("alat") or "alat=" in unit_raw or unit_raw == "":
        local_alat = alat_ang
        m = re.search(r"alat\s*=\s*([0-9EeDd+\-.]+)", unit_raw)
        if m:
            local_alat = float(m.group(1).replace("D", "E").replace("d", "e")) * BOHR_TO_ANG
        if local_alat is None:
            raise QEParseError("CELL_PARAMETERS in alat units but alat was not found in output.")
        return tuple(v_scale(r, local_alat) for r in rows)  # type: ignore[return-value]

    raise QEParseError(f"Unsupported CELL_PARAMETERS unit: '{unit_raw}'")


def _read_atomic_line(line: str) -> tuple[str, Vector] | None:
    parts = line.split()
    if len(parts) < 4:
        return None
    label = parts[0]
    try:
        xyz = (float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError:
        return None
    return label, xyz


def _extract_atoms(
    lines: list[str],
    start_idx: int,
    nat: int | None,
    cell_ang: Matrix3,
    alat_ang: float | None,
) -> list[tuple[str, Vector]]:
    header = lines[start_idx]
    unit_raw = _extract_header_unit(header)

    raw_atoms: list[tuple[str, Vector]] = []
    i = start_idx + 1
    while i < len(lines):
        entry = _read_atomic_line(lines[i])
        if entry is None:
            break
        raw_atoms.append(entry)
        if nat is not None and len(raw_atoms) >= nat:
            break
        i += 1

    if not raw_atoms:
        raise QEParseError("No atoms found in final ATOMIC_POSITIONS block.")

    inv_cell = matrix_inverse_3x3(cell_ang)

    labels = [x[0] for x in raw_atoms]
    coords = [x[1] for x in raw_atoms]

    unit = unit_raw
    if unit.startswith("crystal"):
        frac = coords
    elif unit.startswith("angstrom"):
        frac = [row_vec_times_matrix(c, inv_cell) for c in coords]
    elif unit.startswith("bohr"):
        frac = [row_vec_times_matrix(v_scale(c, BOHR_TO_ANG), inv_cell) for c in coords]
    elif unit.startswith("alat") or unit == "":
        local_alat = alat_ang
        m = re.search(r"alat\s*=\s*([0-9EeDd+\-.]+)", unit)
        if m:
            local_alat = float(m.group(1).replace("D", "E").replace("d", "e")) * BOHR_TO_ANG
        if local_alat is None:
            raise QEParseError("ATOMIC_POSITIONS in alat units but alat was not found in output.")
        frac = [row_vec_times_matrix(v_scale(c, local_alat), inv_cell) for c in coords]
    else:
        raise QEParseError(f"Unsupported ATOMIC_POSITIONS unit: '{unit_raw}'")

    frac_wrapped = [((f[0] % 1.0), (f[1] % 1.0), (f[2] % 1.0)) for f in frac]
    return [(label, frac_wrapped[k]) for k, label in enumerate(labels)]


def _cell_parameters(cell_ang: Matrix3) -> tuple[float, float, float, float, float, float]:
    a_vec, b_vec, c_vec = cell_ang
    a = v_norm(a_vec)
    b = v_norm(b_vec)
    c = v_norm(c_vec)

    def angle(u: Vector, v: Vector) -> float:
        cosang = v_dot(u, v) / (v_norm(u) * v_norm(v))
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))

    alpha = angle(b_vec, c_vec)
    beta = angle(a_vec, c_vec)
    gamma = angle(a_vec, b_vec)
    return a, b, c, alpha, beta, gamma


def _type_symbol_from_label(label: str) -> str:
    m = re.match(r"([A-Za-z]+)", label)
    if not m:
        return "X"
    letters = m.group(1)
    if len(letters) == 1:
        return letters.upper()
    return letters[0].upper() + letters[1:].lower()


def _write_cif(path: Path, cell_ang: Matrix3, atoms_frac: list[tuple[str, Vector]]) -> None:
    a, b, c, alpha, beta, gamma = _cell_parameters(cell_ang)
    lines = [
        "data_qe_final",
        "_symmetry_space_group_name_H-M   'P 1'",
        "_symmetry_Int_Tables_number      1",
        f"_cell_length_a                   {a:.8f}",
        f"_cell_length_b                   {b:.8f}",
        f"_cell_length_c                   {c:.8f}",
        f"_cell_angle_alpha                {alpha:.8f}",
        f"_cell_angle_beta                 {beta:.8f}",
        f"_cell_angle_gamma                {gamma:.8f}",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]

    for i, (label, frac) in enumerate(atoms_frac, start=1):
        type_symbol = _type_symbol_from_label(label)
        atom_label = f"{type_symbol}{i}"
        lines.append(
            f"{atom_label:<8} {type_symbol:<3} {frac[0]: .10f} {frac[1]: .10f} {frac[2]: .10f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_qe_output_to_cif(input_path: Path, output_path: Path) -> None:
    lines = input_path.read_text(encoding="utf-8", errors="replace").splitlines()
    alat_ang = _parse_alat_angstrom(lines)
    nat = _parse_nat(lines)

    cell_idx = _last_block_index(lines, "cell_parameters")
    atom_idx = _last_block_index(lines, "atomic_positions")

    cell_ang = _extract_cell(lines, cell_idx, alat_ang)
    atoms_frac = _extract_atoms(lines, atom_idx, nat, cell_ang, alat_ang)

    _write_cif(output_path, cell_ang, atoms_frac)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a Quantum ESPRESSO pw.x output file, extract the final CELL_PARAMETERS and "
            "ATOMIC_POSITIONS blocks, and write a P1 CIF with wrapped fractional coordinates."
        )
    )
    parser.add_argument("input", type=Path, help="QE output file (pw.x, relax/vc-relax).")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output CIF file path.")
    args = parser.parse_args()

    try:
        convert_qe_output_to_cif(args.input, args.output)
    except QEParseError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
