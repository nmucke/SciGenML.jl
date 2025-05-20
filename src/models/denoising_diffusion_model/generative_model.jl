"""
    ScoreBasedDiffusionModel

A score-based diffusion model generative model.

The score-based diffusion model generative model is a generative model that uses a score-based diffusion model approach to generate data.

It is a special case of the stochastic interpolant generative model where there is no
noise in the interpolant.
"""
mutable struct ScoreBasedDiffusionModel <: Models.GenerativeModel
    interpolant_coefs::Any
    score::Any
    ps::Any
    st::Any
    trait::Any
    device::Any

    function ScoreBasedDiffusionModel(score,)
        score_ps, score_st = Lux.setup(Lux.Random.default_rng(), score)

        ps = (; score = score_ps)
        st = (; score = score_st)
        return new(
            diffusion_interpolant_coefs(),
            score,
            ps,
            st,
            Models.Stochastic(),
            DEFAULT_DEVICE
        )
    end

    ### Constructor from config
    function ScoreBasedDiffusionModel(config::Config.Hyperparameters,)

        # Define score model
        score_model = Architectures.DenseNeuralNetwork(
            config.architecture.in_features,
            config.architecture.out_features,
            config.architecture.hidden_features;
        );
        return ScoreBasedDiffusionModel(score_model)
    end
end
