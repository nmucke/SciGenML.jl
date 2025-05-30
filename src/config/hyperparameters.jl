
"""
    PlaceholderDataHyperparameters

    Hyperparameters for the placeholder.
"""
Configurations.@option "placeholder" struct PlaceholderDataHyperparameters
    num_trajectories::Int
    num_skip_steps::Int
    num_steps::Int
    start_step::Int
end

"""
    Hyperparameters

    Hyperparameters for the architecture, training, and optimizer.
"""
Configurations.@option struct Hyperparameters
    architecture::Union{DenseNeuralNetworkHyperparameters, UNetHyperparameters}
    training::TrainingHyperparameters
    optimizer::OptimizerHyperparameters
    model::Union{
        StochasticInterpolantHyperparameters,
        FollmerStochasticInterpolantHyperparameters,
        FlowMatchingHyperparameters,
        ConditionalFlowMatchingHyperparameters,
        ScoreBasedDiffusionModelHyperparameters
    }
    data::Union{
        KolmogorovDataHyperparameters,
        SuperResKolmogorovDataHyperparameters,
        PlaceholderDataHyperparameters
    }
end
