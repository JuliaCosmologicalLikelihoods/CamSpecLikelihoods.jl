module CamSpecLikelihoods

using Artifacts
using LinearAlgebra
using LinearAlgebra: LAPACK
using NPZ
using CMBForegrounds: PowerLawShape, angular_power
# `loglikelihood` and `predict` are extended from StatsAPI rather than defined
# here. Distributions, StatsBase, DynamicPPL and Turing all extend that same
# binding, so defining rival ones makes `using <this package>, Turing`
# ambiguous. StatsAPI has no dependencies of its own.
import StatsAPI: loglikelihood, predict

export CamSpecPR4Data, CamSpecPR4Parameters, load_camspec_pr4_data
export camspec_pr4_artifact_path
export predict, chi2, loglikelihood

include("data.jl")
include("likelihood.jl")

end
