module CamSpecLikelihoods

using Artifacts
using LinearAlgebra
using LinearAlgebra: LAPACK
using NPZ
using CMBForegrounds: PowerLawShape, angular_power

export CamSpecPR4Data, CamSpecPR4Parameters, load_camspec_pr4_data
export camspec_pr4_artifact_path
export predict, chi2, loglikelihood

include("data.jl")
include("likelihood.jl")

end
