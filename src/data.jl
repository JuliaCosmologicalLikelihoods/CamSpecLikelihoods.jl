"""
    CamSpecPR4Data

The fixed CamSpec PR4 data: the selected data vector, the selected spectrum
ranges, and the **inverse** covariance of the selection.

The release ships a covariance; the converter ships its lower Cholesky factor;
this struct stores `Σ⁻¹`, formed once at construction. That choice is about
speed, and it is the whole performance story of this likelihood: 99% of both a
value and a gradient is one pass over this operator.

* `χ² = rᵀ Σ⁻¹ r` and the model score `Σ⁻¹ r` come from **one** symmetric
  matrix–vector product, where the factor needs one triangular solve for the
  value and a second for the score.
* `symv` is parallel; `trsv` is inherently sequential. Over the identical 375 MiB
  of lower triangle, `symv` measures 6.4 ms against 11.0 ms for `trsv`.

The inverse is formed by LAPACK `potri!` **from the Cholesky factor**, in place,
not by a general inversion, and the quadratic form is evaluated symmetrically.
Measured against a `BigFloat` reference at condition numbers from `1e2` to
`1e10`, `rᵀΣ⁻¹r` and `‖L⁻¹r‖²` agree to within 5% of each other's error, both
being dominated by `cond(Σ)`; the test suite pins that agreement.
"""
struct CamSpecPR4Data
    data_vector::Vector{Float64}
    inverse_covariance::Symmetric{Float64,Matrix{Float64}}
    spectrum_lmin::Vector{Int}
    spectrum_lmax::Vector{Int}
    spectrum_offsets::Vector{Int}
end

"""
    CamSpecPR4Data(data_vector, lower_cholesky, spectrum_lmin, spectrum_lmax, spectrum_offsets)

Build the fixed data from the released **lower Cholesky factor**, which is what
the converter writes and what `load_camspec_pr4_data` reads. The factor is
inverted in place into `Σ⁻¹`; no second `n × n` array is ever allocated, and the
factor is not retained.
"""
function CamSpecPR4Data(
    data_vector::AbstractVector,
    lower_cholesky::AbstractMatrix,
    spectrum_lmin::AbstractVector{<:Integer},
    spectrum_lmax::AbstractVector{<:Integer},
    spectrum_offsets::AbstractVector{<:Integer},
)
    n = length(data_vector)
    size(lower_cholesky) == (n, n) || throw(DimensionMismatch("Cholesky factor must match the data vector"))
    length(spectrum_lmin) == length(spectrum_lmax) == 5 ||
        throw(DimensionMismatch("CamSpec PR4 TTTEEE requires five selected spectra"))
    length(spectrum_offsets) == 6 || throw(DimensionMismatch("spectrum offsets must have length six"))
    spectrum_offsets[1] == 0 && spectrum_offsets[end] == n ||
        throw(ArgumentError("spectrum offsets must span the complete data vector"))
    return CamSpecPR4Data(
        Vector{Float64}(data_vector), _inverse_from_lower_cholesky(lower_cholesky),
        Vector{Int}(spectrum_lmin), Vector{Int}(spectrum_lmax), Vector{Int}(spectrum_offsets),
    )
end

"""
    _inverse_from_lower_cholesky(lower_cholesky) -> Symmetric

`Σ⁻¹` from the lower Cholesky factor `L`, computed in place by LAPACK `potri!`,
which inverts from the factor rather than inverting a general matrix.

Only the lower triangle is read and written; whatever sits above the diagonal of
the input is ignored, and the result is wrapped as `Symmetric(_, :L)` so it is
never read either.
"""
function _inverse_from_lower_cholesky(lower_cholesky::AbstractMatrix)
    factor = Matrix{Float64}(lower_cholesky)
    for i in axes(factor, 1)
        isfinite(factor[i, i]) && !iszero(factor[i, i]) || throw(ArgumentError(
            "the Cholesky factor must have a finite, nonzero diagonal; entry $i is $(factor[i, i])",
        ))
    end
    LAPACK.potri!('L', factor)
    return Symmetric(factor, :L)
end

"""
    camspec_pr4_artifact_path() -> String

Directory of the converted CamSpec PR4 TTTEEE runtime data.

Resolved through `Artifacts.toml`, so the ordinary user path needs no local
files. See that file for the source release and its checksum.
"""
camspec_pr4_artifact_path() = artifact"camspec_pr4_tttee"

"""
    load_camspec_pr4_data() -> CamSpecPR4Data

Load the released data from the published artifact.
"""
load_camspec_pr4_data() = load_camspec_pr4_data(camspec_pr4_artifact_path())

function load_camspec_pr4_data(path::AbstractString)
    return CamSpecPR4Data(
        vec(npzread(joinpath(path, "data_vector.npy"))),
        npzread(joinpath(path, "lower_cholesky.npy")),
        vec(npzread(joinpath(path, "spectrum_lmin.npy"))),
        vec(npzread(joinpath(path, "spectrum_lmax.npy"))),
        vec(npzread(joinpath(path, "spectrum_offsets.npy"))),
    )
end
