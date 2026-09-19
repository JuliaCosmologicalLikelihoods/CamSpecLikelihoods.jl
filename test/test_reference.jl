@testset "CamSpec PR4 Cobaya prediction fixtures" begin
    fixture = joinpath(@__DIR__, "fixtures", "camspec_pr4", "camspec_pr4_reference.npz")
    values = npzread(fixture)
    lmins = [30, 500, 500, 30, 30]
    lmaxs = [2000, 2500, 2500, 2000, 2000]
    offsets = cumsum([0; lmaxs .- lmins .+ 1])
    for name in ("baseline", "foregrounds", "calibration")
        parameters = CamSpecPR4Parameters(
            values["$(name)__A_planck"], values["$(name)__cal2"],
            values["$(name)__calTE"], values["$(name)__calEE"],
            (values["$(name)__amp_143"], values["$(name)__amp_217"], values["$(name)__amp_143x217"]),
            (values["$(name)__n_143"], values["$(name)__n_217"], values["$(name)__n_143x217"]),
        )
        model = CamSpecLikelihoods._predict(
            lmins, lmaxs, offsets, values["DlTT"], values["DlTE"], values["DlEE"], parameters,
        )
        @test model ≈ values["$(name)_prediction"] rtol=1e-13 atol=1e-13
    end
end

@testset "CamSpec PR4 chi2 against the frozen Cobaya reference" begin
    # Until the artifact existed, nothing in this package had ever loaded the
    # released 9915x9915 covariance: only `predict` was checked against Cobaya,
    # and the chi2/loglike in reference.json sat unasserted. This closes that.
    data = load_camspec_pr4_data()
    @test length(data.data_vector) == 9915
    @test data.spectrum_lmin == [30, 500, 500, 30, 30]
    @test data.spectrum_lmax == [2000, 2500, 2500, 2000, 2000]
    @test data.spectrum_offsets == [0, 1971, 3972, 5973, 7944, 9915]

    fixture = joinpath(@__DIR__, "fixtures", "camspec_pr4", "camspec_pr4_reference.npz")
    values = npzread(fixture)
    reference = read(joinpath(@__DIR__, "fixtures", "camspec_pr4", "reference.json"), String)
    metric(name, key) = parse(Float64, match(
        Regex("\"$name\"\\s*:\\s*\\{[^}]*\"$key\"\\s*:\\s*(-?[0-9.eE+]+)"), reference)[1])

    for name in ("baseline", "foregrounds", "calibration")
        parameters = CamSpecPR4Parameters(
            values["$(name)__A_planck"], values["$(name)__cal2"],
            values["$(name)__calTE"], values["$(name)__calEE"],
            (values["$(name)__amp_143"], values["$(name)__amp_217"], values["$(name)__amp_143x217"]),
            (values["$(name)__n_143"], values["$(name)__n_217"], values["$(name)__n_143x217"]))
        model = predict(data, values["DlTT"], values["DlTE"], values["DlEE"], parameters)

        # The prediction still matches to 1e-13; the quadratic form agrees with
        # Cobaya to 5.7e-8, consistently across all three points and in the same
        # direction. That residual is Cobaya's own arithmetic on the float32
        # release covariance against this float64 factorization: it is NOT the
        # inverse-covariance formulation, which agrees with the triangular-solve
        # route to 2.8e-15 on this operator (cond = 5.5e8). rtol is set an order
        # of magnitude above the observed offset, tight enough to catch a real
        # regression, and it must not be loosened to hide one.
        @test model ≈ values["$(name)_prediction"] rtol=1e-13 atol=1e-13
        @test chi2(data, model) ≈ metric(name, "chi2") rtol=1e-6
        @test loglikelihood(data, model) ≈ metric(name, "loglike") rtol=1e-6
        @test loglikelihood(data, model) ≈ -chi2(data, model) / 2
    end
end
