"""
    test/test_turing.jl

The Turing extension. The likelihood is stated with `~`, not `@addlogprob!`, so
these tests check what that choice buys: the distribution's `logpdf` is this
package's own likelihood plus the Gaussian normalization, the model's log joint
is exactly priors plus likelihood, and the normalization itself is the real
constant rather than a plausible-looking number.

Unlike the ACT extension, CamSpec's `~` is normalized: `log_normalization` is
taken from the Cholesky diagonal before `potri!` destroys the factor. The
synthetic testset below is the one that can check that against an explicit
`MvNormal`, because at 9915 x 9915 the released operator is far too large to
form a reference Gaussian from.
"""

using ADTypes
using Turing
using Distributions
using Random

const TURING_EXT = Base.get_extension(CamSpecLikelihoods, :CamSpecLikelihoodsTuringExt)

# A small, exactly-known Gaussian, built the same way test_ad.jl builds its
# synthetic cases: five spectra of two bandpowers each. Defined here rather
# than reused from another test file, so this file does not depend on include
# order.
function synthetic_camspec_data()
    n = 10
    covariance = Matrix(2.0 * I(n))
    for i in 1:n, j in 1:n
        covariance[i, j] += 0.05 * sin(i + j)
    end
    lower_cholesky = cholesky(Symmetric(covariance, :L)).L
    data = CamSpecPR4Data(
        collect(1.0:10.0), lower_cholesky,
        [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [0, 2, 4, 6, 8, 10],
    )
    return data, Symmetric(covariance)
end

const SYNTHETIC_DlTT = collect(1.0:3.0) ./ 3
const SYNTHETIC_DlTE = collect(1.0:3.0) ./ 6
const SYNTHETIC_DlEE = collect(1.0:3.0) ./ 9

@testset "CamSpec Turing extension — loaded" begin
    @test TURING_EXT !== nothing
end

@testset "CamSpec Turing extension — the `~` is a normalized Gaussian" begin
    data, covariance = synthetic_camspec_data()
    model = [0.9, 1.8, 3.1, 4.2, 4.8, 6.3, 6.9, 8.4, 9.1, 9.7]
    d = TURING_EXT.CamSpecPR4Bandpowers(data, model)

    @test length(d) == 10

    # The whole point of `log_normalization`: this must agree with an explicit
    # Gaussian, not merely differ from it by a constant. If the normalization
    # were dropped, mis-signed, or computed from Σ⁻¹ instead of Σ, this fails.
    reference = MvNormal(model, covariance)
    @test logpdf(d, data.data_vector) ≈ logpdf(reference, data.data_vector) rtol=1e-12

    # Correct at a second point too, which a constant offset bug would survive.
    other = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0]
    @test logpdf(d, other) ≈ logpdf(reference, other) rtol=1e-12

    # And the normalization is the documented constant on its own.
    @test gaussian_normalization(data) ≈
        -10 / 2 * log(2 * pi) - logdet(covariance) / 2 rtol=1e-12
end

@testset "CamSpec Turing extension — logpdf is the package likelihood" begin
    data = load_camspec_pr4_data()
    values = npzread(joinpath(@__DIR__, "fixtures", "camspec_pr4", "camspec_pr4_reference.npz"))
    parameters = CamSpecPR4Parameters(
        values["baseline__A_planck"], values["baseline__cal2"],
        values["baseline__calTE"], values["baseline__calEE"],
        (values["baseline__amp_143"], values["baseline__amp_217"], values["baseline__amp_143x217"]),
        (values["baseline__n_143"], values["baseline__n_217"], values["baseline__n_143x217"]),
    )
    model = predict(data, values["DlTT"], values["DlTE"], values["DlEE"], parameters)
    d = TURING_EXT.CamSpecPR4Bandpowers(data, model)

    @test length(d) == 9915
    # Exactly, not approximately: the distribution contracts the same inverse
    # covariance `chi2` does, so the only difference from `loglikelihood` is the
    # normalization this package keeps separate.
    @test logpdf(d, data.data_vector) ==
        loglikelihood(data, model) + gaussian_normalization(data)

    # A different vector must give a different, finite answer: the distribution
    # must not ignore its argument and echo `data.data_vector`.
    shifted = data.data_vector .+ 1.0
    @test isfinite(logpdf(d, shifted))
    @test logpdf(d, shifted) != logpdf(d, data.data_vector)

    @test_throws DimensionMismatch TURING_EXT.CamSpecPR4Bandpowers(data, model[1:10])

    # Sampling is refused deliberately — the package stores Σ⁻¹, not a factor of
    # Σ, so a draw would mean factorizing the 9915 x 9915 operator. This is the
    # opposite of the ACT extension, where `rand` works, and it is a documented
    # choice rather than an omission.
    @test_throws ArgumentError rand(Random.MersenneTwister(20260920), d)
end

@testset "CamSpec Turing extension — priors are the documented defaults" begin
    priors = TURING_EXT.CAMSPEC_PR4_DEFAULT_PRIORS
    @test length(priors) == 10
    @test keys(priors) == (:A_planck, :cal2, :calTE, :calEE,
                           :amp_143, :amp_217, :amp_143x217,
                           :n_143, :n_217, :n_143x217)

    # These are wide, *unofficial* defaults: CamSpec's released configuration is
    # not reproduced in this package. Pinning them exactly is what keeps that
    # claim honest — a silent edit here would quietly change everyone's
    # posterior while the docstring still said "documented default".
    @test priors.A_planck == Normal(1.0, 0.0025)
    @test priors.cal2 == Normal(1.0, 0.01)
    @test priors.calTE == Normal(1.0, 0.01)
    @test priors.calEE == Normal(1.0, 0.01)
    @test priors.amp_143 == Uniform(0.0, 50.0)
    @test priors.amp_217 == Uniform(0.0, 100.0)
    @test priors.amp_143x217 == Uniform(0.0, 50.0)
    @test priors.n_143 == Uniform(0.0, 3.0)
    @test priors.n_217 == Uniform(0.0, 3.0)
    @test priors.n_143x217 == Uniform(0.0, 3.0)
end

@testset "CamSpec Turing extension — log joint is priors plus likelihood" begin
    data, _ = synthetic_camspec_data()
    model_fn = TURING_EXT.camspec_pr4_model(
        data, SYNTHETIC_DlTT, SYNTHETIC_DlTE, SYNTHETIC_DlEE,
    )
    values = (A_planck = 1.0, cal2 = 1.0, calTE = 1.0, calEE = 1.0,
              amp_143 = 10.0, amp_217 = 20.0, amp_143x217 = 10.0,
              n_143 = 1.0, n_217 = 1.0, n_143x217 = 1.0)

    parameters = CamSpecPR4Parameters(
        values.A_planck, values.cal2, values.calTE, values.calEE,
        (values.amp_143, values.amp_217, values.amp_143x217),
        (values.n_143, values.n_217, values.n_143x217),
    )
    prediction = predict(data, SYNTHETIC_DlTT, SYNTHETIC_DlTE, SYNTHETIC_DlEE, parameters)
    likelihood_term = loglikelihood(data, prediction) + gaussian_normalization(data)
    priors = TURING_EXT.CAMSPEC_PR4_DEFAULT_PRIORS
    prior_term = sum(logpdf(priors[name], values[name]) for name in keys(priors))

    @test Turing.DynamicPPL.logjoint(model_fn, values) ≈
        prior_term + likelihood_term rtol=1e-12
    @test isfinite(Turing.DynamicPPL.logjoint(model_fn, values))
end

@testset "CamSpec Turing extension — NUTS samples" begin
    # The synthetic case, not the released one: this asserts that the model is
    # differentiable and that the sampler can move through it, which the 9915 x
    # 9915 operator would establish no better at many times the cost.
    data, _ = synthetic_camspec_data()
    model_fn = TURING_EXT.camspec_pr4_model(
        data, SYNTHETIC_DlTT, SYNTHETIC_DlTE, SYNTHETIC_DlEE,
    )
    start = (A_planck = 1.0, cal2 = 1.0, calTE = 1.0, calEE = 1.0,
             amp_143 = 10.0, amp_217 = 20.0, amp_143x217 = 10.0,
             n_143 = 1.0, n_217 = 1.0, n_143x217 = 1.0)
    Random.seed!(20260920)
    # `initial_params` is not optional in practice: `cal2` enters as sqrt(cal2)
    # for the 143x217 calibration, so a draw from its unconstrained Normal prior
    # can land on a negative value and throw instead of rejecting.
    chain = sample(model_fn, NUTS(2, 0.65; adtype=AutoForwardDiff()), 2;
                   initial_params=start, progress=false, verbose=false)
    @test size(chain, 1) == 2
    parameters = [k for k in keys(chain) if occursin("Parameter", string(k))]
    @test length(parameters) == 10
end
