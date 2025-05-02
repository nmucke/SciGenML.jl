using Test

@testitem "Interpolant Coefs" begin
    import SciGenML.Models as Models

    @testset "Linear Interpolant" begin
        # Test deterministic linear interpolant coefficients
        interpolant = Models.linear_interpolant_coefs(Models.Deterministic())

        # Test alpha and beta functions
        t = 0.5f0
        @test interpolant.alpha(t) ≈ 0.5f0
        @test interpolant.beta(t) ≈ 0.5f0

        # Test derivatives
        @test interpolant.alpha_diff(t) ≈ -1.0f0
        @test interpolant.beta_diff(t) ≈ 1.0f0

        # Test boundary conditions
        @test interpolant.alpha(0.0f0) ≈ 1.0f0
        @test interpolant.alpha(1.0f0) ≈ 0.0f0
        @test interpolant.beta(0.0f0) ≈ 0.0f0
        @test interpolant.beta(1.0f0) ≈ 1.0f0

        # Test stochastic linear interpolant coefficients
        stoch_interpolant = Models.linear_interpolant_coefs(Models.Stochastic())

        # Test gamma function
        @test stoch_interpolant.gamma(t) ≈ sqrt(2.0f0 * t * (1.0f0 - t))
        @test stoch_interpolant.gamma(0.0f0) ≈ 0.0f0
        @test stoch_interpolant.gamma(1.0f0) ≈ 0.0f0

        # Test gamma derivative
        @test stoch_interpolant.gamma_diff(t) ≈
              (1.0f0 - 2.0f0 * t) / (sqrt(2.0f0) * sqrt(-(t - 1.0f0) * t) + Models.ZERO_TOL)
    end

    @testset "Quadratic Interpolant" begin
        # Test deterministic quadratic interpolant coefficients
        interpolant = Models.quadratic_interpolant_coefs(Models.Deterministic())

        # Test alpha and beta functions
        t = 0.5f0
        @test interpolant.alpha(t) ≈ 0.5f0
        @test interpolant.beta(t) ≈ 0.25f0

        # Test derivatives
        @test interpolant.alpha_diff(t) ≈ -1.0f0
        @test interpolant.beta_diff(t) ≈ 1.0f0

        # Test boundary conditions
        @test interpolant.alpha(0.0f0) ≈ 1.0f0
        @test interpolant.alpha(1.0f0) ≈ 0.0f0
        @test interpolant.beta(0.0f0) ≈ 0.0f0
        @test interpolant.beta(1.0f0) ≈ 1.0f0

        # Test stochastic quadratic interpolant coefficients
        stoch_interpolant = Models.quadratic_interpolant_coefs(Models.Stochastic())

        # Test gamma function
        @test stoch_interpolant.gamma(t) ≈ sqrt(2.0f0 * t * (1.0f0 - t))
        @test stoch_interpolant.gamma(0.0f0) ≈ 0.0f0
        @test stoch_interpolant.gamma(1.0f0) ≈ 0.0f0

        # Test gamma derivative
        @test stoch_interpolant.gamma_diff(t) ≈
              (1.0f0 - 2.0f0 * t) / (sqrt(2.0f0) * sqrt(-(t - 1.0f0) * t) + Models.ZERO_TOL)
    end

    @testset "Compute Interpolant" begin
        # Test compute_interpolant function
        x0 = [1.0f0, 2.0f0]
        x1 = [3.0f0, 4.0f0]
        z = [0.1f0, 0.2f0]
        t = 0.5f0

        # Test with deterministic linear interpolant
        linear = Models.linear_interpolant_coefs(Models.Deterministic())
        result = Models.compute_interpolant(x0, x1, t, linear)
        expected = [2.0f0, 3.0f0]  # (x0 + x1) / 2
        @test result ≈ expected

        # Test with stochastic linear interpolant
        stoch_linear = Models.linear_interpolant_coefs(Models.Stochastic())
        result = Models.compute_interpolant(x0, x1, z, t, stoch_linear)
        expected = [2.0f0, 3.0f0] .+ stoch_linear.gamma(t) .* z
        @test result ≈ expected

        # Test with deterministic quadratic interpolant
        quadratic = Models.quadratic_interpolant_coefs(Models.Deterministic())
        result = Models.compute_interpolant(x0, x1, t, quadratic)
        expected = [1.25f0, 2.0f0]  # x0 + 0.25 * (x1 - x0)
        @test result ≈ expected

        # Test with stochastic quadratic interpolant
        stoch_quadratic = Models.quadratic_interpolant_coefs(Models.Stochastic())
        result = Models.compute_interpolant(x0, x1, z, t, stoch_quadratic)
        expected = [1.25f0, 2.0f0] .+ stoch_quadratic.gamma(t) .* z
        @test result ≈ expected
    end

    @testset "Compute Interpolant Derivative" begin
        # Test compute_interpolant_diff function
        x0 = [1.0f0, 2.0f0]
        x1 = [3.0f0, 4.0f0]
        z = [0.1f0, 0.2f0]
        t = 0.5f0

        # Test with deterministic linear interpolant
        linear = Models.linear_interpolant_coefs(Models.Deterministic())
        result = Models.compute_interpolant_diff(x0, x1, t, linear)
        expected = [2.0f0, 2.0f0]  # x1 - x0
        @test result ≈ expected

        # Test with stochastic linear interpolant
        stoch_linear = Models.linear_interpolant_coefs(Models.Stochastic())
        result = Models.compute_interpolant_diff(x0, x1, z, t, stoch_linear)
        expected = [2.0f0, 2.0f0] .+ stoch_linear.gamma_diff(t) .* z
        @test result ≈ expected

        # Test with deterministic quadratic interpolant
        quadratic = Models.quadratic_interpolant_coefs(Models.Deterministic())
        result = Models.compute_interpolant_diff(x0, x1, t, quadratic)
        expected = [2.0f0, 2.0f0]  # -x0 + (2t)*x1 at t=0.5
        @test result ≈ expected

        # Test with stochastic quadratic interpolant
        stoch_quadratic = Models.quadratic_interpolant_coefs(Models.Stochastic())
        result = Models.compute_interpolant_diff(x0, x1, z, t, stoch_quadratic)
        expected = [2.0f0, 2.0f0] .+ stoch_quadratic.gamma_diff(t) .* z
        @test result ≈ expected
    end
end
