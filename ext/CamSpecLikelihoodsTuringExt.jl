"""
    CamSpecLikelihoodsTuringExt

Turing.jl support for the CamSpec PR4 TTTEEE likelihood.

Loaded automatically when `CamSpecLikelihoods`, `Turing` and `Distributions` are
all present. It provides [`CamSpecPR4Bandpowers`](@ref), a `Distribution` over
the 9915 selected bandpowers, so a model states the likelihood with `~`:

```julia
data.data_vector ~ CamSpecPR4Bandpowers(data, model)
```

rather than injecting a number with `@addlogprob!`. `~` keeps the observation
visible to DynamicPPL, so conditioning, prior sampling and log-density
decompositions all behave normally.

`logpdf` contracts the precomputed inverse covariance — the same operation
`chi2` performs — so the model's likelihood term is exactly

    loglikelihood(data, model) + gaussian_normalization(data)

and reverse mode stays on the registered `_fixed_inverse_apply` rule. Going
through `MvNormalCanon` instead would solve `μ = J \\ h` on every evaluation and
store the precision matrix twice.
"""
module CamSpecLikelihoodsTuringExt

using CamSpecLikelihoods
using Turing
using Distributions
using LinearAlgebra
using Random

# CamSpecLikelihoods extends the StatsAPI bindings for `predict` and
# `loglikelihood`, which Turing also extends, so importing them explicitly here
# keeps the references unambiguous inside this module.
import CamSpecLikelihoods: _fixed_inverse_apply, predict, loglikelihood

export CamSpecPR4Bandpowers, camspec_pr4_model

"""
    CamSpecPR4Bandpowers(data, model)

The CamSpec PR4 bandpower likelihood as a multivariate distribution: a Gaussian
with the released fixed covariance and mean `model`.

The covariance is released data, never a parameter, so it is carried by
reference and its precomputed inverse is reused on every evaluation.
"""
struct CamSpecPR4Bandpowers{D<:CamSpecPR4Data,M<:AbstractVector} <:
       ContinuousMultivariateDistribution
    data::D
    model::M

    function CamSpecPR4Bandpowers(data::CamSpecPR4Data, model::AbstractVector)
        length(model) == length(data.data_vector) || throw(DimensionMismatch(
            "model must have $(length(data.data_vector)) entries, got $(length(model))",
        ))
        return new{typeof(data), typeof(model)}(data, model)
    end
end

Base.length(d::CamSpecPR4Bandpowers) = length(d.data.data_vector)
Base.eltype(::Type{<:CamSpecPR4Bandpowers{<:Any,M}}) where {M} = eltype(M)

function Distributions._logpdf(d::CamSpecPR4Bandpowers, x::AbstractVector{<:Real})
    residual = x .- d.model
    quadratic = dot(residual, _fixed_inverse_apply(d.data.inverse_covariance, residual))
    return -quadratic / 2 + d.data.log_normalization
end

# Unlike the ACT likelihood, this package stores the inverse covariance and not
# a factor of the covariance, so there is no cheap square root to sample with.
# Drawing would need a Cholesky of Σ⁻¹ and a solve per draw; rather than hide
# that cost behind `rand`, say so.
function Distributions._rand!(::Random.AbstractRNG, ::CamSpecPR4Bandpowers,
                              ::AbstractVector{<:Real})
    throw(ArgumentError(
        "CamSpecPR4Bandpowers cannot be sampled from: the package stores Σ⁻¹ " *
        "rather than a factor of Σ, so a draw would require factorizing the " *
        "9915 x 9915 operator. Use logpdf for inference; build an explicit " *
        "MvNormal from the released Cholesky factor if you need draws.",
    ))
end

"""
    camspec_pr4_model(data, DlTT, DlTE, DlEE; priors = CAMSPEC_PR4_DEFAULT_PRIORS)

Turing model for the CamSpec PR4 TTTEEE likelihood: the ten nuisance parameters
and then the bandpower likelihood, all through `~`.

The theory spectra are held fixed; sample over cosmology by passing different
spectra per evaluation or by wrapping this model in a larger one.

**The priors are not official.** CamSpec's released configuration is not
reproduced in this package, so `CAMSPEC_PR4_DEFAULT_PRIORS` is a deliberately
wide, documented default rather than a published prescription. Pass your own.
"""
@model function camspec_pr4_model(data::CamSpecPR4Data,
                                  DlTT::AbstractVector,
                                  DlTE::AbstractVector,
                                  DlEE::AbstractVector;
                                  priors = CAMSPEC_PR4_DEFAULT_PRIORS)
    A_planck ~ priors.A_planck
    cal2 ~ priors.cal2
    calTE ~ priors.calTE
    calEE ~ priors.calEE
    amp_143 ~ priors.amp_143
    amp_217 ~ priors.amp_217
    amp_143x217 ~ priors.amp_143x217
    n_143 ~ priors.n_143
    n_217 ~ priors.n_217
    n_143x217 ~ priors.n_143x217

    parameters = CamSpecPR4Parameters(
        A_planck, cal2, calTE, calEE,
        (amp_143, amp_217, amp_143x217),
        (n_143, n_217, n_143x217),
    )
    model = predict(data, DlTT, DlTE, DlEE, parameters)

    data.data_vector ~ CamSpecPR4Bandpowers(data, model)

    return parameters
end

"""
    CAMSPEC_PR4_DEFAULT_PRIORS

Wide, **unofficial** defaults for the ten CamSpec PR4 nuisance parameters, so
[`camspec_pr4_model`](@ref) is runnable out of the box.

`A_planck` carries the usual Planck absolute-calibration prior. The calibration
ratios sit near unity and the TT residual amplitudes and tilts are given ranges
that comfortably contain the released posterior. None of this is a published
prescription: replace it with the configuration you intend to sample.
"""
const CAMSPEC_PR4_DEFAULT_PRIORS = (
    A_planck = Normal(1.0, 0.0025),
    cal2 = Normal(1.0, 0.01),
    calTE = Normal(1.0, 0.01),
    calEE = Normal(1.0, 0.01),
    amp_143 = Uniform(0.0, 50.0),
    amp_217 = Uniform(0.0, 100.0),
    amp_143x217 = Uniform(0.0, 50.0),
    n_143 = Uniform(0.0, 3.0),
    n_217 = Uniform(0.0, 3.0),
    n_143x217 = Uniform(0.0, 3.0),
)

end # module
