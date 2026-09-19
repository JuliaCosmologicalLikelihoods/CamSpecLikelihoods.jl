module CamSpecLikelihoodsChainRulesCoreExt

using CamSpecLikelihoods: _fixed_inverse_apply, _loglikelihood_parts, _nuisance_cotangents,
    _parameters_from_vector, CamSpecPR4Data, CamSpecPR4Parameters
using CamSpecLikelihoods: loglikelihood
using LinearAlgebra: Symmetric
import ChainRulesCore
using ChainRulesCore: NoTangent, ProjectTo, unthunk

# Σ⁻¹ is released data and symmetric, so the operator takes `NoTangent()` and
# the pullback is the same product applied to the incoming cotangent.
function ChainRulesCore.rrule(
    ::typeof(_fixed_inverse_apply),
    inverse_covariance::Symmetric,
    residual::AbstractVector,
)
    weighted = _fixed_inverse_apply(inverse_covariance, residual)
    project_residual = ProjectTo(residual)

    function fixed_inverse_apply_pullback(weighted_bar_thunked)
        weighted_bar = unthunk(weighted_bar_thunked)
        residual_bar = project_residual(inverse_covariance * weighted_bar)
        return NoTangent(), NoTangent(), residual_bar
    end

    return weighted, fixed_inverse_apply_pullback
end

# Whole-likelihood analytical pullback over the flat parameter vector:
# no tape entries for assembly or covariance solves.
function ChainRulesCore.rrule(
    ::typeof(loglikelihood),
    data::CamSpecPR4Data,
    DlTT::AbstractVector,
    DlTE::AbstractVector,
    DlEE::AbstractVector,
    x::AbstractVector{<:Real},
)
    parameters = _parameters_from_vector(x)
    model, score, calibrations, chi2_value = _loglikelihood_parts(data, DlTT, DlTE, DlEE, parameters)
    logL = -chi2_value / 2

    function loglikelihood_pullback(logL_bar_thunked)
        logL_bar = unthunk(logL_bar_thunked)
        # The incoming cotangent must scale every returned one. Discarding it
        # happens to be invisible for a plain `gradient`, where the seed is 1,
        # and silently wrong for anything that composes this likelihood — a
        # tempered or weighted posterior, or any nonlinear function of logL.
        #
        # Every cotangent `_nuisance_cotangents` produces is linear in `score`,
        # so seeding the score applies the scaling exactly, in one vector
        # multiply rather than nine.
        seeded_score = logL_bar .* score
        dA, dcal2, dcalTE, dcalEE, damplitudes, dtilts, dDlTT, dDlTE, dDlEE = _nuisance_cotangents(
            data, DlTT, DlTE, DlEE, parameters, seeded_score, calibrations, model,
        )
        x_bar = [dA, dcal2, dcalTE, dcalEE, damplitudes..., dtilts...]
        return NoTangent(), NoTangent(), dDlTT, dDlTE, dDlEE, x_bar
    end

    return logL, loglikelihood_pullback
end

end
