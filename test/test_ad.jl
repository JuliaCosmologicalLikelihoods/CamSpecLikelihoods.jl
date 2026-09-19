@testset "CamSpec PR4 fixed-covariance Mooncake rule" begin
    lower_cholesky = [2.0 0.0; 0.5 sqrt(3.0)]
    data = CamSpecPR4Data(
        [1.2, -0.3], lower_cholesky, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 2, 2, 2, 2],
    )
    objective = model -> loglikelihood(data, model)
    model = [0.4, -0.8]
    forwarddiff = DifferentiationInterface.gradient(objective, AutoForwardDiff(), model)
    mooncake = DifferentiationInterface.gradient(objective, AutoMooncake(; config=nothing), model)
    @test all(isfinite, mooncake)
    @test mooncake ≈ forwarddiff rtol=1e-12
end

@testset "CamSpec PR4 analytical whole-likelihood Mooncake rule" begin
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
    DlTT = collect(1.0:3.0) ./ 3
    DlTE = collect(1.0:3.0) ./ 6
    DlEE = collect(1.0:3.0) ./ 9
    objective = x -> loglikelihood(data, DlTT, DlTE, DlEE, x)
    x0 = [1.0, 1.0, 1.0, 1.0, 10.0, 20.0, 10.0, 1.0, 1.0, 1.0]
    forwarddiff = DifferentiationInterface.gradient(objective, AutoForwardDiff(), x0)
    mooncake = DifferentiationInterface.gradient(objective, AutoMooncake(; config=nothing), x0)
    @test all(isfinite, mooncake)
    @test mooncake ≈ forwarddiff rtol=1e-10

    # theory-spectrum cotangents must also be correct
    function theory_objective(dltt)
        return loglikelihood(
            data, dltt, DlTE, DlEE,
            CamSpecPR4Parameters(1.0, 1.0, 1.0, 1.0, (10.0, 20.0, 10.0), (1.0, 1.0, 1.0)),
        )
    end
    fd_theory = DifferentiationInterface.gradient(theory_objective, AutoForwardDiff(), DlTT)
    mc_theory = DifferentiationInterface.gradient(theory_objective, AutoMooncake(; config=nothing), DlTT)
    @test all(isfinite, mc_theory)
    @test mc_theory ≈ fd_theory rtol=1e-10
end

@testset "CamSpec PR4 pullback honours arbitrary cotangents" begin
    # The whole-likelihood rule must scale its cotangents by the incoming one.
    # A plain `gradient` seeds it with 1, so a rule that drops the seed looks
    # correct there and is silently wrong for every composition: a tempered or
    # weighted posterior, or any nonlinear function of the likelihood.
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
    DlTT = collect(1.0:3.0) ./ 3
    DlTE = collect(1.0:3.0) ./ 6
    DlEE = collect(1.0:3.0) ./ 9
    x0 = [1.0, 1.0, 1.0, 1.0, 10.0, 20.0, 10.0, 1.0, 1.0, 1.0]

    # 1. The rule itself: the pullback is linear in the seed, exactly.
    _, pullback = ChainRulesCore.rrule(
        CamSpecLikelihoods.loglikelihood, data, DlTT, DlTE, DlEE, x0,
    )
    unit = pullback(1.0)
    for seed in (2.5, -3.0, 0.0, 1e-8)
        scaled = pullback(seed)
        for (index, name) in ((3, "DlTT"), (4, "DlTE"), (5, "DlEE"), (6, "parameters"))
            @test scaled[index] ≈ seed .* unit[index] rtol=1e-12 atol=1e-14
        end
    end
    # A thunked seed, as Mooncake passes it, must behave identically.
    thunked = pullback(ChainRulesCore.Thunk(() -> 2.5))
    @test thunked[6] ≈ 2.5 .* unit[6] rtol=1e-12

    # 2. End to end through Mooncake, against ForwardDiff, for objectives that
    #    do not seed with 1.
    forward = AutoForwardDiff()
    reverse = AutoMooncake(; config=nothing)
    base(x) = loglikelihood(data, DlTT, DlTE, DlEE, x)
    for objective in (x -> 3.0 * base(x), x -> base(x)^2, x -> exp(base(x) / 100))
        expected = DifferentiationInterface.gradient(objective, forward, x0)
        actual = DifferentiationInterface.gradient(objective, reverse, x0)
        @test all(isfinite, actual)
        @test actual ≈ expected rtol=1e-8
    end

    # The unit-seed case must be unchanged by the fix.
    @test DifferentiationInterface.gradient(base, reverse, x0) ≈
        DifferentiationInterface.gradient(base, forward, x0) rtol=1e-10
end
