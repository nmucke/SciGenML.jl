using Test

@testitem "Interpolant Coefs" begin
    import SciGenML.Models as Models

    @testset "Linear Interpolant" begin
        # Test linear interpolant coefficients
        interpolant = Models.linear_interpolant_coefs()

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
    end

    @testset "Quadratic Interpolant" begin
        # Test quadratic interpolant coefficients
        interpolant = Models.quadratic_interpolant_coefs()

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
    end

    @testset "Compute Interpolant" begin
        # Test compute_interpolant function
        x0 = [1.0f0, 2.0f0]
        x1 = [3.0f0, 4.0f0]
        t = 0.5f0

        # Test with linear interpolant
        linear = Models.linear_interpolant_coefs()
        result = Models.compute_interpolant(x0, x1, linear, t)
        expected = [2.0f0, 3.0f0]  # (x0 + x1) / 2
        @test result ≈ expected

        # Test with quadratic interpolant
        quadratic = Models.quadratic_interpolant_coefs()
        result = Models.compute_interpolant(x0, x1, quadratic, t)
        expected = [1.25f0, 2.0f0]  # x0 + 0.25 * (x1 - x0)
        @test result ≈ expected
    end

    @testset "Compute Interpolant Derivative" begin
        # Test compute_interpolant_diff function
        x0 = [1.0f0, 2.0f0]
        x1 = [3.0f0, 4.0f0]
        t = 0.5f0

        # Test with linear interpolant
        linear = Models.linear_interpolant_coefs()
        result = Models.compute_interpolant_diff(x0, x1, linear, t)
        expected = [2.0f0, 2.0f0]  # x1 - x0
        @test result ≈ expected

        # Test with quadratic interpolant
        quadratic = Models.quadratic_interpolant_coefs()
        result = Models.compute_interpolant_diff(x0, x1, quadratic, t)
        expected = [2.0f0, 2.0f0]  # -x0 + (2t)*x1 at t=0.5
        @test result ≈ expected
    end
end
