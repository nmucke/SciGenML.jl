
"""
    TrainingHyperparameters

    Hyperparameters for the training.
"""
Configurations.@option struct TrainingHyperparameters
    batch_size::Int
    num_epochs::Int
    match_base_and_target::Bool
    patience::Int
end

"""
    OptimizerHyperparameters

    Hyperparameters for the optimizer.
"""
Configurations.@option struct OptimizerHyperparameters
    type::String
    learning_rate::DEFAULT_TYPE
    weight_decay::DEFAULT_TYPE

    # CONSTRUCTOR
    function OptimizerHyperparameters(type, learning_rate, weight_decay)
        return new(type, learning_rate |> DEFAULT_TYPE, weight_decay |> DEFAULT_TYPE)
    end
end
