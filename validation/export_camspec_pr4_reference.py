#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np

from cobaya.likelihoods.planck_NPIPE_highl_CamSpec.TTTEEE import TTTEEE


def prediction(likelihood, dl_tt, dl_te, dl_ee, parameters):
    calibrations = likelihood.get_cals(parameters)
    foregrounds = likelihood.get_foregrounds(parameters)
    blocks = []
    for index, size in enumerate(likelihood.used_sizes):
        if size == 0:
            continue
        ells = np.asarray(likelihood.ell_ranges[index])
        theory = dl_tt if index <= 3 else dl_te if index == 4 else dl_ee
        block = theory[ells].copy()
        if index <= 3:
            block += foregrounds[index, ells]
        blocks.append(block / calibrations[index])
    return np.concatenate(blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    likelihood = TTTEEE(packages_path=str(args.packages_path))
    ells = np.arange(np.max(likelihood.lmax) + 1, dtype=np.int64)
    dl_tt = 5500.0 * np.exp(-ells / 1450.0) + 180.0 * np.sin(ells / 85.0) ** 2
    dl_te = 115.0 * np.exp(-ells / 1750.0) * np.cos(ells / 105.0)
    dl_ee = 48.0 * np.exp(-ells / 1800.0) + 4.0 * np.sin(ells / 70.0) ** 2

    baseline = {
        "use_fg_residual_model": 0,
        "A_planck": 1.0,
        "cal0": 1.0,
        "cal2": 1.0,
        "calTE": 1.0,
        "calEE": 1.0,
        "amp_100": 0.0,
        "amp_143": 10.0,
        "amp_217": 20.0,
        "amp_143x217": 10.0,
        "n_100": 1.0,
        "n_143": 1.0,
        "n_217": 1.0,
        "n_143x217": 1.0,
    }
    cases = {
        "baseline": {},
        "foregrounds": {
            "amp_143": 13.2,
            "amp_217": 24.1,
            "amp_143x217": 8.4,
            "n_143": 0.82,
            "n_217": 1.21,
            "n_143x217": 1.08,
        },
        "calibration": {
            "A_planck": 1.0017,
            "cal2": 0.994,
            "calTE": 1.006,
            "calEE": 0.991,
        },
    }

    arrays = {"ell": ells, "DlTT": dl_tt, "DlTE": dl_te, "DlEE": dl_ee}
    metrics = {}
    for name, updates in cases.items():
        parameters = baseline | updates
        model = prediction(likelihood, dl_tt, dl_te, dl_ee, parameters)
        chi2 = likelihood.chi_squared(dl_tt, dl_te, dl_ee, parameters)
        arrays[f"{name}_prediction"] = model
        metrics[name] = {"chi2": float(chi2), "loglike": float(-chi2 / 2)}
        for parameter, value in parameters.items():
            arrays[f"{name}__{parameter}"] = np.asarray(value)

    np.savez_compressed(args.output / "camspec_pr4_reference.npz", **arrays)
    (args.output / "reference.json").write_text(
        json.dumps(
            {
                "cobaya_version": "3.5.6",
                "target": "planck_NPIPE_highl_CamSpec.TTTEEE",
                "data_archive_sha256": "1978dd0c148a87678b4cf07340401814a3bc81b5cb87dbdccb76565a3a883e15",
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
