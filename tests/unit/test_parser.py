"""Unit tests for raw Boltzmann data table parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dmra.cosmology.ede.parser import (
    build_background_history_from_table,
    parse_class_power_spectrum,
    parse_class_table,
)


def test_parse_class_table_and_background(tmp_path: Path) -> None:
    """Test parsing a synthetic CLASS background table and building BackgroundHistory."""
    bg_file = tmp_path / "background.dat"
    # Create sample CLASS background data
    lines = [
        "# 1:z 2:H [1/Mpc] 3:omega_m 4:omega_r 5:omega_lambda 6:omega_ede\n",
        "0.0 0.000225 0.315 0.0001 0.6849 0.0\n",
        "1.0 0.000450 0.750 0.0005 0.2495 0.0\n",
        "10.0 0.003000 0.980 0.0190 0.0000 0.001\n",
    ]
    bg_file.write_text("".join(lines), encoding="utf-8")

    col_map, data = parse_class_table(bg_file)
    assert "z" in col_map
    assert "h" in col_map
    assert data.shape == (3, 6)

    bg_hist = build_background_history_from_table(data, col_map)
    assert len(bg_hist.z) == 3
    assert bg_hist.H_z[0] > 0.0
    assert np.all(bg_hist.omega_m > 0.0)


def test_parse_class_power_spectrum(tmp_path: Path) -> None:
    """Test parsing a synthetic CLASS pk.dat file with unit conversion."""
    pk_file = tmp_path / "pk.dat"
    lines = [
        "# k (1/Mpc) P (Mpc^3)\n",
        "0.01 1000.0\n",
        "0.10 5000.0\n",
        "1.00 200.0\n",
    ]
    pk_file.write_text("".join(lines), encoding="utf-8")

    h = 0.70
    k_h, pk_h3 = parse_class_power_spectrum(pk_file, h=h)

    # In DMRA units: k_h = k_mpc / h, pk_h3 = pk_mpc3 * h^3
    np.testing.assert_allclose(k_h, np.array([0.01, 0.10, 1.00]) / h)
    np.testing.assert_allclose(pk_h3, np.array([1000.0, 5000.0, 200.0]) * (h**3))


def test_parse_table_file_not_found() -> None:
    """Test parse_class_table raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        parse_class_table("/nonexistent/path/to/table.dat")


def test_parse_table_empty_file(tmp_path: Path) -> None:
    """Test parse_class_table raises ValueError for empty file."""
    empty_file = tmp_path / "empty.dat"
    empty_file.write_text("# only comment line\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no numeric data rows"):
        parse_class_table(empty_file)
