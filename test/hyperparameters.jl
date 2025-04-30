using Test

@testitem "Hyperparameters" begin
    using SciGenML.Config
    import Configurations

    @testset "ModelHyperparameters" begin
        # Test basic construction with scalar in_features
        hp = DenseNeuralNetworkHyperparameters(10, 2, [64, 32], 0.5)
        @test hp.in_features == 10
        @test hp.out_features == 2
        @test hp.hidden_features == [64, 32]
        @test hp.dropout == 0.5

        # Test construction with vector in_features (should convert to tuple)
        hp = DenseNeuralNetworkHyperparameters([10, 20], 2, [64, 32], 0.5)
        @test hp.in_features == (10, 20)
        @test typeof(hp.in_features) == Tuple{Int64, Int64}
    end

    @testset "TrainingHyperparameters" begin
        hp = TrainingHyperparameters(32, 100)
        @test hp.batch_size == 32
        @test hp.num_epochs == 100
    end

    @testset "OptimizerHyperparameters" begin
        hp = OptimizerHyperparameters(0.001f0, 0.0001f0)
        @test hp.learning_rate == 0.001f0
        @test hp.weight_decay == 0.0001f0
    end

    @testset "Complete Hyperparameters" begin
        model_hp = DenseNeuralNetworkHyperparameters(10, 2, [64, 32], 0.5)
        training_hp = TrainingHyperparameters(32, 100)
        optimizer_hp = OptimizerHyperparameters(0.001f0, 0.0001f0)

        hp = Hyperparameters(model_hp, training_hp, optimizer_hp)

        @test hp.model == model_hp
        @test hp.training == training_hp
        @test hp.optimizer == optimizer_hp
    end

    @testset "Load config" begin
        using SciGenML.Config
        config = Configurations.from_toml(
            Config.Hyperparameters,
            "dense_neural_network_config.toml"
        )

        @test config.model.in_features == (20, 10)
        @test config.model.out_features == 1
        @test config.model.hidden_features == [10, 10]
        @test config.model.dropout == 0.1

        @test config.training.batch_size == 32
        @test config.training.num_epochs == 100

        @test config.optimizer.learning_rate == 0.001f0
        @test config.optimizer.weight_decay == 0.001f0
    end
end
