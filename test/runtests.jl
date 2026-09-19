using CamSpecLikelihoods
using ADTypes: AutoForwardDiff, AutoMooncake
using ChainRulesCore
using DifferentiationInterface
using ForwardDiff
using LinearAlgebra
using Mooncake
using NPZ
using Test

include("test_core.jl")
include("test_reference.jl")
include("test_ad.jl")
