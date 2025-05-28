
"""
    FlowMatchingHyperparameters

    Hyperparameters for the flow matching generative model.
"""
Configurations.@option "flow_matching" struct FlowMatchingHyperparameters
    interpolant_type::String
end

"""
    ConditionalFlowMatchingHyperparameters

    Hyperparameters for the conditional flow matching generative model.
"""
Configurations.@option "conditional_flow_matching" struct ConditionalFlowMatchingHyperparameters
    interpolant_type::String
    guidance_scale::DEFAULT_TYPE
    replacement_probability::DEFAULT_TYPE
    unconditional_condition::DEFAULT_TYPE
end

"""
    StochasticInterpolantHyperparameters

    Hyperparameters for the stochastic interpolant generative model.
"""
Configurations.@option "stochastic_interpolant" struct StochasticInterpolantHyperparameters
    interpolant_type::String
end

"""
    FollmerStochasticInterpolantHyperparameters

    Hyperparameters for the follmer stochastic interpolant generative model.
"""
Configurations.@option "follmer_stochastic_interpolant" struct FollmerStochasticInterpolantHyperparameters
    interpolant_type::String
end

"""
    ScoreBasedDiffusionModelHyperparameters

    Hyperparameters for the score-based diffusion model generative model.
"""
Configurations.@option "score_based_diffusion_model" struct ScoreBasedDiffusionModelHyperparameters
    interpolant_type::String
end
