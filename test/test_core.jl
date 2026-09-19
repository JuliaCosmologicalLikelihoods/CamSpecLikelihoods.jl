@testset "CamSpec PR4 Gaussian core" begin
    data = CamSpecPR4Data(
        [1.0, 2.0, 3.0, 4.0, 5.0], Matrix{Float64}(I, 5, 5),
        [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], collect(0:5),
    )
    parameters = CamSpecPR4Parameters(1.0, 1.0, 1.0, 1.0, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    model = predict(data, [1.0], [4.0], [5.0], parameters)
    @test model == [1.0, 1.0, 1.0, 4.0, 5.0]
    @test chi2(data, model) == 5.0
    @test loglikelihood(data, model) == -2.5
end

@testset "CamSpec PR4 inverse covariance matches the triangular-solve route" begin
    # The likelihood stores Σ⁻¹ rather than the Cholesky factor it is built
    # from, because one symmetric product yields both the chi-square and the
    # model score. That is only legitimate if it agrees with the factor it
    # replaced, so compare against an explicit triangular solve.
    rng_values = [sin(3.0 * i) for i in 1:60]
    n = 12
    covariance = Matrix(3.0 * I(n))
    for i in 1:n, j in 1:n
        covariance[i, j] += 0.4 * cos(1.7 * (i + 2j))
    end
    covariance = Symmetric(covariance, :L)
    factor = cholesky(covariance).L
    offsets = [0, 3, 6, 8, 10, n]
    data = CamSpecPR4Data(
        rng_values[1:n], factor, [0, 0, 0, 0, 0], [2, 2, 1, 1, 1], offsets,
    )
    model = rng_values[21:(20 + n)]

    residual = data.data_vector .- model
    whitened = factor \ residual
    @test chi2(data, model) ≈ dot(whitened, whitened) rtol=1e-12
    @test loglikelihood(data, model) ≈ -dot(whitened, whitened) / 2 rtol=1e-12

    # Σ⁻¹ really is the inverse of the covariance the factor came from.
    @test data.inverse_covariance * (covariance * residual) ≈ residual rtol=1e-10
    @test data.inverse_covariance ≈ inv(Matrix(covariance)) rtol=1e-10

    # The stored operator is symmetric and carries no factor.
    @test data.inverse_covariance isa Symmetric{Float64,Matrix{Float64}}
    @test !hasproperty(data, :lower_cholesky)

    # A factor with a zero on the diagonal cannot be inverted and must say so.
    singular = Matrix(factor); singular[4, 4] = 0.0
    @test_throws ArgumentError CamSpecPR4Data(
        rng_values[1:n], singular, [0, 0, 0, 0, 0], [2, 2, 1, 1, 1], offsets,
    )
end
