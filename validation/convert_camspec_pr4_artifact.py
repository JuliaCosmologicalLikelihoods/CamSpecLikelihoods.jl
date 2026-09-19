#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg


SELECTED = ("143x143", "217x217", "143x217", "TE", "EE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ranges = []
    for line in (args.source / "like_NPIPE_12.6_unified_data_ranges.txt").read_text(
        encoding="utf-8-sig"
    ).splitlines():
        if line.strip():
            name, lmin, lmax = line.split()
            ranges.append((name, int(lmin), int(lmax)))

    spectra = np.loadtxt(args.source / "like_NPIPE_12.6_unified_spectra.txt")
    full_indices = []
    data_blocks = []
    offset = 0
    selected_ranges = []
    for column, (name, lmin, lmax) in enumerate(ranges):
        size = lmax - lmin + 1
        if name in SELECTED:
            full_indices.append(np.arange(offset, offset + size, dtype=np.int64))
            data_blocks.append(spectra[lmin : lmax + 1, column])
            selected_ranges.append((name, lmin, lmax))
        offset += size
    indices = np.concatenate(full_indices)
    data_vector = np.concatenate(data_blocks).astype(np.float64)

    covariance_file = args.source / "like_NPIPE_12.6_unified_cov.bin"
    dimension = int(np.sqrt(covariance_file.stat().st_size // np.dtype(np.float32).itemsize))
    if dimension != offset:
        raise ValueError(f"covariance dimension {dimension} does not match ranges {offset}")
    full_covariance = np.memmap(
        covariance_file, dtype=np.float32, mode="r", shape=(dimension, dimension)
    )
    covariance = np.asarray(full_covariance[np.ix_(indices, indices)], dtype=np.float64)
    lower_cholesky = scipy.linalg.cholesky(
        covariance, lower=True, overwrite_a=True, check_finite=False
    )

    np.save(args.output / "data_vector.npy", data_vector)
    np.save(args.output / "lower_cholesky.npy", lower_cholesky)
    np.save(args.output / "spectrum_lmin.npy", np.asarray([item[1] for item in selected_ranges]))
    np.save(args.output / "spectrum_lmax.npy", np.asarray([item[2] for item in selected_ranges]))
    np.save(
        args.output / "spectrum_offsets.npy",
        np.cumsum([0] + [item[2] - item[1] + 1 for item in selected_ranges]),
    )
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "spectrum_order": [item[0] for item in selected_ranges],
                "data_length": int(data_vector.size),
                "release_covariance_dtype": "float32",
                "stored_factor_dtype": "float64",
                "source_archive_sha256": "1978dd0c148a87678b4cf07340401814a3bc81b5cb87dbdccb76565a3a883e15",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
