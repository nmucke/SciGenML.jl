using Test

@testitem "StochasticInterpolant" begin
    import SciGenML.Models as Models
    import SciGenML.Config as Config
    import SciGenML.Architectures as Architectures
    using Lux

    @testset "Constructor with velocity and score" begin
        # Create simple neural networks for testing
        velocity = Architectures.DenseNeuralNetwork(2, 2, [4])
        score = Architectures.DenseNeuralNetwork(2, 2, [4])

        # Test constructor
        model = Models.StochasticInterpolant(velocity, score)

        @test model.velocity == velocity
        @test model.score == score
        @test model.trait == Models.Stochastic()
        @test !isnothing(model.ps.velocity)
        @test !isnothing(model.ps.score)
        @test !isnothing(model.st.velocity)
        @test !isnothing(model.st.score)
    end

    @testset "Constructor with interpolant type" begin
        # Create simple neural networks for testing
        velocity = Architectures.DenseNeuralNetwork(2, 2, [4])
        score = Architectures.DenseNeuralNetwork(2, 2, [4])

        # Test constructor with linear interpolant
        model = Models.StochasticInterpolant("linear", velocity, score)

        @test model.velocity == velocity
        @test model.score == score
        @test model.trait == Models.Stochastic()
        @test !isnothing(model.ps.velocity)
        @test !isnothing(model.ps.score)
        @test !isnothing(model.st.velocity)
        @test !isnothing(model.st.score)
    end

    @testset "Deterministic constructor" begin
        # Create simple neural network for testing
        velocity = Architectures.DenseNeuralNetwork(2, 2, [4])

        # Test constructor
        model = Models.StochasticInterpolant(velocity)

        @test model.velocity == velocity
        @test isnothing(model.score)
        @test model.trait == Models.Deterministic()
        @test !isnothing(model.ps.velocity)
        @test !isnothing(model.st.velocity)
    end

    @testset "Deterministic constructor with interpolant type" begin
        # Create simple neural network for testing
        velocity = Architectures.DenseNeuralNetwork(2, 2, [4])

        # Test constructor with linear interpolant
        model = Models.StochasticInterpolant("linear", velocity)

        @test model.velocity == velocity
        @test isnothing(model.score)
        @test model.trait == Models.Deterministic()
        @test !isnothing(model.ps.velocity)
        @test !isnothing(model.st.velocity)
    end

    @testset "Constructor from config" begin
        # Create a basic config
        architecture_hp = Config.DenseNeuralNetworkHyperparameters(2, 2, [4], 0.1)
        training_hp = Config.TrainingHyperparameters(32, 100)
        optimizer_hp = Config.OptimizerHyperparameters("adam", 0.001f0, 0.0001f0)
        model_hp = Config.StochasticInterpolantHyperparameters("linear")
        config =
            Config.Hyperparameters(architecture_hp, training_hp, optimizer_hp, model_hp)

        # Test stochastic constructor
        model = Models.StochasticInterpolant(config)

        @test model.trait == Models.Stochastic()
        @test !isnothing(model.velocity)
        @test !isnothing(model.score)
        @test !isnothing(model.ps.velocity)
        @test !isnothing(model.ps.score)
        @test !isnothing(model.st.velocity)
        @test !isnothing(model.st.score)

        # Test deterministic constructor
        model = Models.StochasticInterpolant(config, Models.Deterministic())

        @test model.trait == Models.Deterministic()
        @test !isnothing(model.velocity)
        @test isnothing(model.score)
        @test !isnothing(model.ps.velocity)
        @test !isnothing(model.st.velocity)
    end
end
