# CamSpec PR4 native Julia implementation

## Frozen target

The first target is Cobaya's native Planck NPIPE CamSpec `TTTEEE` likelihood:

- likelihood: `planck_NPIPE_highl_CamSpec.TTTEEE`;
- data release: `CobayaSampler/planck_native_data`, release `v1`, asset `CamSpec_NPIPE.zip`;
- archive SHA-256: `1978dd0c148a87678b4cf07340401814a3bc81b5cb87dbdccb76565a3a883e15`;
- selected spectra: `143x143`, `217x217`, `143x217`, `TE`, `EE`;
- selected data-vector length: 9915;
- theory convention: externally supplied `D_ell` in microkelvin squared, matching Cobaya's `ell_factor=true` interface.

The default PR4 target uses the CamSpec 2021 power-law residual model
(`use_fg_residual_model = 0`). The older physical foreground-template model is
not part of the first vertical slice.

## Reference implementations

1. Cobaya 3.5.6 native CamSpec engine:
   `cobaya.likelihoods.base_classes.planck_2018_CamSpec_python` plus
   `planck_2018_highl_CamSpec2021` and the NPIPE subclass.
2. Local direct-spectrum translation:
   `/home/marcobonici/Desktop/work/CosmologicalEmulators/jax-loglike/jax_loglike/camspec_pr4.py`.

Cobaya is authoritative. The JAX translation is a useful cross-check, but its
existing tests are smoke tests rather than numerical reference fixtures.

## Ordered implementation

1. Export reproducible Cobaya fixtures at a fixed theory/nuisance point.
2. Convert the release to a numeric Julia artifact containing the selected data
   vector, spectrum ranges, and a lower covariance Cholesky factor. Do not retain
   both the selected covariance and its factor in memory.
3. Implement the fixed-data Gaussian core and verify the quadratic form.
4. Implement the three TT residual power laws and TT/TE/EE calibration mapping.
5. Keep Gaussian and uniform priors separate from the data likelihood.
6. Validate baseline and multipoint predictions against Cobaya, then compare
   Mooncake, ForwardDiff, and finite-difference gradients through
   DifferentiationInterface.
7. Add the TT/TE/EE subset constructors only after TTTEEE parity is complete.

## Numerical pressure points

- The released covariance is Float32 and the selected TTTEEE block is
  9915 x 9915. Preserve the release quantization, promote to Float64 for the
  factorization, store only the lower factor, and use a fixed-data no-tangent
  solve primitive.
- CamSpec predicts `D_ell`, not `C_ell`, at the selected multipoles.
- Calibration factors are
  `[cal0, 1, cal2, sqrt(cal2), calTE, calEE] * A_planck^2`; the selected PR4
  TTTEEE configuration does not use `100x100` or `cal0`.
- The baseline TT residuals are independent power laws at pivot ell=1500 for
  `143x143`, `217x217`, and `143x217`.
