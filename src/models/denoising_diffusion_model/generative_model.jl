"""
    ScoreBasedDiffusionModel

A score-based diffusion model generative model.

The score-based diffusion model generative model is a generative model that uses a score-based diffusion model approach to generate data.

It is a special case of the stochastic interpolant generative model where there is no
noise in the interpolant.
"""
mutable struct ScoreBasedDiffusionModel <: Models.GenerativeModel
    interpolant_coefs::Any
    velocity::Any
    ps::Any
    st::Any
    trait::Any
    device::Any

    function ScoreBasedDiffusionModel(velocity,)
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)

        ps = (; velocity = velocity_ps)
        st = (; velocity = velocity_st)
        return new(
            diffusion_interpolant_coefs(5.0f0),
            velocity,
            ps,
            st,
            Models.Stochastic(),
            DEFAULT_DEVICE
        )
    end

    ### Constructor from config
    function ScoreBasedDiffusionModel(config::Config.Hyperparameters,)

        # Define score model
        velocity_model = Architectures.get_architecture(config.architecture);

        return ScoreBasedDiffusionModel(velocity_model)
    end
end

function drift_term(model::ScoreBasedDiffusionModel, diffusion_fn)
    return drift_term(model.trait, model, diffusion_fn)
end

"""
    drift_term(
        ::Models.Stochastic,
        model::StochasticInterpolant, 
        diffusion_fn, 
    )

    Compute the drift term for a stochsatic interpolant.
"""
function drift_term(::Models.Stochastic, model::ScoreBasedDiffusionModel, diffusion_fn)
    function drift_wrapper(x, ps, st; model = model)
        velocity, _velocity_st = model.velocity(x, ps.velocity, st.velocity)
        st = (; velocity = _velocity_st)

        t = x[end]
        diffusion = diffusion_fn(t)
        diffusion = Utils.reshape_scalar(diffusion, ndims(x[1]))

        alpha = model.interpolant_coefs.alpha(t)
        alpha = Utils.reshape_scalar(alpha, ndims(x[1]))

        alpha_diff = model.interpolant_coefs.alpha_diff(t)
        alpha_diff = Utils.reshape_scalar(alpha_diff, ndims(x[1]))

        beta = model.interpolant_coefs.beta(t)
        beta = Utils.reshape_scalar(beta, ndims(x[1]))

        beta_diff = model.interpolant_coefs.beta_diff(t)
        beta_diff = Utils.reshape_scalar(beta_diff, ndims(x[1]))

        score_numerator = beta .* velocity - beta_diff .* x[1]

        score_denominator = alpha .^ 2 .* beta_diff - beta .* alpha_diff .* alpha
        score_denominator = score_denominator .+ ZERO_TOL

        score_term = score_numerator ./ score_denominator

        return velocity + 0.5f0 .* diffusion .^ 2 .* score_term, st
    end

    return drift_wrapper
end
