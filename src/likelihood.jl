struct CamSpecPR4Parameters{T}
    A_planck::T
    cal2::T
    calTE::T
    calEE::T
    amplitudes::NTuple{3,T}
    tilts::NTuple{3,T}
end

function CamSpecPR4Parameters(A_planck, cal2, calTE, calEE, amplitudes::NTuple{3}, tilts::NTuple{3})
    values = promote(A_planck, cal2, calTE, calEE, amplitudes..., tilts...)
    T = typeof(values[1])
    return CamSpecPR4Parameters{T}(
        values[1], values[2], values[3], values[4],
        (values[5], values[6], values[7]), (values[8], values[9], values[10]),
    )
end

function predict(
    data::CamSpecPR4Data,
    DlTT::AbstractVector,
    DlTE::AbstractVector,
    DlEE::AbstractVector,
    parameters::CamSpecPR4Parameters,
)
    return _predict(
        data.spectrum_lmin, data.spectrum_lmax, data.spectrum_offsets,
        DlTT, DlTE, DlEE, parameters,
    )
end

function _predict(
    spectrum_lmin::AbstractVector{<:Integer},
    spectrum_lmax::AbstractVector{<:Integer},
    spectrum_offsets::AbstractVector{<:Integer},
    DlTT::AbstractVector,
    DlTE::AbstractVector,
    DlEE::AbstractVector,
    parameters::CamSpecPR4Parameters,
)
    maximum(spectrum_lmax) < minimum((length(DlTT), length(DlTE), length(DlEE))) ||
        throw(DimensionMismatch("theory vectors must include ell=0 through the maximum selected multipole"))
    calibrations = (
        parameters.A_planck^2,
        parameters.cal2 * parameters.A_planck^2,
        sqrt(parameters.cal2) * parameters.A_planck^2,
        parameters.calTE * parameters.A_planck^2,
        parameters.calEE * parameters.A_planck^2,
    )
    return _assemble_model(
        spectrum_lmin, spectrum_lmax, spectrum_offsets,
        DlTT, DlTE, DlEE, calibrations, parameters.amplitudes, parameters.tilts,
    )
end

function _assemble_model(
    spectrum_lmin::AbstractVector{<:Integer},
    spectrum_lmax::AbstractVector{<:Integer},
    spectrum_offsets::AbstractVector{<:Integer},
    DlTT::AbstractVector,
    DlTE::AbstractVector,
    DlEE::AbstractVector,
    calibrations::NTuple{5},
    amplitudes::NTuple{3},
    tilts::NTuple{3},
)
    T = promote_type(eltype(DlTT), eltype(DlTE), eltype(DlEE), typeof(calibrations[1]))
    model = Vector{T}(undef, spectrum_offsets[end])
    residual_shape = PowerLawShape(1500.0)
    for spectrum in 1:5
        lmin = spectrum_lmin[spectrum]
        lmax = spectrum_lmax[spectrum]
        output = @view model[spectrum_offsets[spectrum]+1:spectrum_offsets[spectrum+1]]
        theory = spectrum <= 3 ? DlTT : spectrum == 4 ? DlTE : DlEE
        output .= @view(theory[lmin+1:lmax+1])
        if spectrum <= 3
            output .+= angular_power(residual_shape, lmin:lmax, tilts[spectrum]; amp=amplitudes[spectrum])
        end
        output ./= calibrations[spectrum]
    end
    return model
end

"""
    _fixed_inverse_apply(inverse_covariance, residual) -> Σ⁻¹ r

The one expensive operation in this package, isolated as a reverse-mode
primitive. `Σ⁻¹` is released data, so it carries no tangent; the operator is
symmetric, so the rule is self-adjoint.
"""
@inline _fixed_inverse_apply(inverse_covariance::Symmetric, residual::AbstractVector) =
    inverse_covariance * residual

function chi2(data::CamSpecPR4Data, model::AbstractVector)
    length(model) == length(data.data_vector) || throw(DimensionMismatch("model and data vectors differ"))
    residual = data.data_vector .- model
    return dot(residual, _fixed_inverse_apply(data.inverse_covariance, residual))
end

loglikelihood(data::CamSpecPR4Data, model::AbstractVector) = -chi2(data, model) / 2

function loglikelihood(
    data::CamSpecPR4Data,
    DlTT::AbstractVector,
    DlTE::AbstractVector,
    DlEE::AbstractVector,
    parameters::CamSpecPR4Parameters,
)
    return loglikelihood(data, predict(data, DlTT, DlTE, DlEE, parameters))
end

# Analytical pullback of the whole likelihood. Avoids recording the model
# assembly and the covariance application on the AD tape: the model score
#     s = Σ^{-1} r  (one symmetric matrix-vector product)
# is contracted with the assembly Jacobian in closed form. The same product
# yields the chi-square, as dot(r, s), so a gradient costs exactly one pass over
# the operator — the same as a bare likelihood value.

function _loglikelihood_parts(
    data::CamSpecPR4Data,
    DlTT::AbstractVector,
    DlTE::AbstractVector,
    DlEE::AbstractVector,
    parameters::CamSpecPR4Parameters,
)
    calibrations = (
        parameters.A_planck^2,
        parameters.cal2 * parameters.A_planck^2,
        sqrt(parameters.cal2) * parameters.A_planck^2,
        parameters.calTE * parameters.A_planck^2,
        parameters.calEE * parameters.A_planck^2,
    )
    model = _assemble_model(
        data.spectrum_lmin, data.spectrum_lmax, data.spectrum_offsets,
        DlTT, DlTE, DlEE, calibrations, parameters.amplitudes, parameters.tilts,
    )
    residual = data.data_vector .- model
    # d logL / d model = Σ^{-1} r, and chi2 = dot(r, Σ^{-1} r): one product.
    score = _fixed_inverse_apply(data.inverse_covariance, residual)
    chi2_value = dot(residual, score)
    return model, score, calibrations, chi2_value
end

function _nuisance_cotangents(
    data::CamSpecPR4Data,
    DlTT::AbstractVector,
    DlTE::AbstractVector,
    DlEE::AbstractVector,
    parameters::CamSpecPR4Parameters,
    score::AbstractVector,
    calibrations::NTuple{5},
    model::AbstractVector,
)
    T = promote_type(eltype(score), typeof(calibrations[1]))
    dA = zero(T); dcal2 = zero(T); dcalTE = zero(T); dcalEE = zero(T)
    damplitudes = (zero(T), zero(T), zero(T))
    dtilts = (zero(T), zero(T), zero(T))
    dDlTT = zeros(T, length(DlTT))
    dDlTE = zeros(T, length(DlTE))
    dDlEE = zeros(T, length(DlEE))
    residual_shape = PowerLawShape(1500.0)
    for spectrum in 1:5
        block = @view score[data.spectrum_offsets[spectrum]+1:data.spectrum_offsets[spectrum+1]]
        c = calibrations[spectrum]
        lmin = data.spectrum_lmin[spectrum]
        lmax = data.spectrum_lmax[spectrum]
        ells = lmin:lmax
        theory_bar = spectrum <= 3 ? dDlTT : spectrum == 4 ? dDlTE : dDlEE
        # d model / d theory = 1 / c
        theory_bar[lmin+1:lmax+1] .+= block ./ c
        # d logL / d c = -dot(block, model) / c
        dlogL_dc = -dot(block, @view(model[data.spectrum_offsets[spectrum]+1:data.spectrum_offsets[spectrum+1]])) / c
        if spectrum <= 3
            shape = angular_power(residual_shape, ells, parameters.tilts[spectrum])
            damplitudes = Base.setindex(damplitudes, damplitudes[spectrum] + dot(block, shape) / c, spectrum)
            dtilts = Base.setindex(
                dtilts,
                dtilts[spectrum] +
                parameters.amplitudes[spectrum] * dot(block, shape .* log.(ells ./ 1500.0)) / c,
                spectrum,
            )
        end
        # c1 = A^2; c2 = cal2 A^2; c3 = sqrt(cal2) A^2; c4 = calTE A^2; c5 = calEE A^2
        if spectrum == 1
            dA += dlogL_dc * 2 * parameters.A_planck
        elseif spectrum == 2
            dA += dlogL_dc * 2 * parameters.cal2 * parameters.A_planck
            dcal2 += dlogL_dc * parameters.A_planck^2
        elseif spectrum == 3
            dA += dlogL_dc * 2 * sqrt(parameters.cal2) * parameters.A_planck
            dcal2 += dlogL_dc * parameters.A_planck^2 / (2 * sqrt(parameters.cal2))
        elseif spectrum == 4
            dA += dlogL_dc * 2 * parameters.calTE * parameters.A_planck
            dcalTE += dlogL_dc * parameters.A_planck^2
        else
            dA += dlogL_dc * 2 * parameters.calEE * parameters.A_planck
            dcalEE += dlogL_dc * parameters.A_planck^2
        end
    end
    return dA, dcal2, dcalTE, dcalEE, damplitudes, dtilts, dDlTT, dDlTE, dDlEE
end

_parameters_from_vector(x::AbstractVector) = CamSpecPR4Parameters(
    x[1], x[2], x[3], x[4], (x[5], x[6], x[7]), (x[8], x[9], x[10]),
)

function loglikelihood(
    data::CamSpecPR4Data,
    DlTT::AbstractVector,
    DlTE::AbstractVector,
    DlEE::AbstractVector,
    x::AbstractVector{<:Real},
)
    return loglikelihood(data, DlTT, DlTE, DlEE, _parameters_from_vector(x))
end
