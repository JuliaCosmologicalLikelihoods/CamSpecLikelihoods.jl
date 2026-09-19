module CamSpecLikelihoodsMooncakeExt

using CamSpecLikelihoods: _fixed_inverse_apply, loglikelihood, CamSpecPR4Data
using LinearAlgebra: Symmetric
using Mooncake: @from_chainrules, MinimalCtx
import Mooncake

# Release data is never a differentiable quantity. Without this, Mooncake
# builds (and re-zeroes every call) a full 9915^2 tangent for the inverse
# covariance field — ~780 MB of useless memory traffic per gradient.
Mooncake.tangent_type(::Type{CamSpecPR4Data}) = Mooncake.NoTangent

@from_chainrules MinimalCtx Tuple{
    typeof(_fixed_inverse_apply), Symmetric{Float64,Matrix{Float64}}, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(loglikelihood), CamSpecPR4Data, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64},
}

end
