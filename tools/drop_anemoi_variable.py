#!/usr/bin/env python3

"""
Utility for dropping a single variable from an Anemoi dataset stored in zarr format.

The datasets produced by ``anemoi-datasets`` keep every meteorological variable as
one index along the ``variable`` dimension inside the ``data`` array instead of
owning one array per variable.  Because of that, a plain ``Dataset.drop_vars`` call
does not work – you need to slice the ``variable`` dimension and update its metadata.

Usage examples:
    WindPower/bin/python tools/drop_anemoi_variable.py synthetic_windpower data/Cerra_Anemoids.zarr
    WindPower/bin/python tools/drop_anemoi_variable.py synthetic_windpower data/Cerra_inner_Anemoids.zarr --in-place
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import xarray as xr


def _update_list_attr(attrs: dict, field: str, to_remove: str) -> None:
    """Remove `to_remove` from a list attribute if it is present."""
    if field in attrs and isinstance(attrs[field], Iterable):
        attrs[field] = [value for value in attrs[field] if value != to_remove]


def drop_variable_from_store(
    store_path: Path,
    variable_name: str,
    output_path: Path | None,
    *,
    consolidated: bool,
    in_place: bool,
    keep_backup: bool,
) -> None:
    if not store_path.exists():
        raise FileNotFoundError(f"{store_path} does not exist")

    ds = xr.open_zarr(store_path, consolidated=consolidated)
    variables: List[str] = list(ds.attrs.get("variables", []))
    if not variables:
        raise ValueError(f"{store_path} does not look like an Anemoi zarr store")
    if variable_name not in variables:
        raise ValueError(f"{variable_name} not present in {store_path}")

    keep_idx = [idx for idx, name in enumerate(variables) if name != variable_name]
    trimmed = ds.isel(variable=keep_idx)
    attrs = dict(trimmed.attrs)

    attrs["variables"] = [variables[idx] for idx in keep_idx]
    metadata = dict(attrs.get("variables_metadata", {}))
    metadata.pop(variable_name, None)
    attrs["variables_metadata"] = metadata

    for field_name in ("constant_fields", "variables_with_nans"):
        _update_list_attr(attrs, field_name, variable_name)

    trimmed.attrs = attrs

    if output_path is None:
        output_path = store_path.with_name(f"{store_path.name}_drop_{variable_name}")

    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists")

    trimmed.to_zarr(output_path, mode="w", consolidated=consolidated)

    if in_place:
        backup_path = store_path.with_name(f"{store_path.name}.bak")
        if backup_path.exists():
            raise FileExistsError(f"Backup path {backup_path} already exists")

        store_path.rename(backup_path)
        output_path.rename(store_path)

        if not keep_backup:
            shutil.rmtree(backup_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop a variable from one or more Anemoi zarr datasets."
    )
    parser.add_argument(
        "variable",
        help="Name of the variable to drop (e.g. synthetic_windpower).",
    )
    parser.add_argument(
        "stores",
        nargs="+",
        help="Paths to the *.zarr stores that should be processed.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional output store path. Only allowed when a single store is provided. "
            "By default a sibling '<store>_drop_<variable>' directory is created."
        ),
    )
    parser.add_argument(
        "--consolidated",
        action="store_true",
        help="Treat the input stores as consolidated metadata zarr stores.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the original store with the trimmed one (keeps a .bak folder).",
    )
    parser.add_argument(
        "--discard-backup",
        action="store_true",
        help="Remove the backup folder when --in-place is used.",
    )

    args = parser.parse_args()
    variable_name: str = args.variable
    store_paths = [Path(store_path) for store_path in args.stores]

    if args.output and len(store_paths) > 1:
        raise ValueError("--output can only be used with a single store")

    for store_path in store_paths:
        drop_variable_from_store(
            store_path,
            variable_name,
            Path(args.output) if args.output else None,
            consolidated=args.consolidated,
            in_place=args.in_place,
            keep_backup=not args.discard_backup,
        )


if __name__ == "__main__":
    main()
