"""
    StochasticInterpolant

A stochastic interpolant generative model.
"""
mutable struct StochasticInterpolant <: GenerativeModel
    interpolant_coefs::Any
    velocity::Any
    score::Any
    ps::Any
    st::Any
    trait::Any

    ### Stochastic sampling interpolant
    # Constructor with velocity and score
    function StochasticInterpolant(velocity, score)
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)
        score_ps, score_st = Lux.setup(Lux.Random.default_rng(), score)

        ps = (; velocity = velocity_ps, score = score_ps)
        st = (; velocity = velocity_st, score = score_st)
        return new(
            linear_interpolant_coefs(Models.Stochastic()),
            velocity,
            score,
            ps,
            st,
            Models.Stochastic()
        )
    end

    # Constructor with interpolant type
    function StochasticInterpolant(interpolant_type::String, velocity, score)
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)
        score_ps, score_st = Lux.setup(Lux.Random.default_rng(), score)

        ps = (; velocity = velocity_ps, score = score_ps)
        st = (; velocity = velocity_st, score = score_st)
        return new(
            get_interpolant_coefs(Models.Stochastic(), interpolant_type),
            velocity,
            score,
            ps,
            st,
            Models.Stochastic()
        )
    end

    ### Deterministic sampling interpolant
    # Constructor for deterministic interpolant
    function StochasticInterpolant(velocity,)
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)
        ps = (; velocity = velocity_ps)
        st = (; velocity = velocity_st)
        return new(
            linear_interpolant_coefs(Models.Stochastic()),
            velocity,
            nothing,
            ps,
            st,
            Models.Deterministic()
        )
    end

    ### Constructor for stochastic interpolant
    function StochasticInterpolant(interpolant_type::String, velocity)
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)
        ps = (; velocity = velocity_ps)
        st = (; velocity = velocity_st)
        return new(
            get_interpolant_coefs(Models.Stochastic(), interpolant_type),
            velocity,
            nothing,
            ps,
            st,
            Models.Deterministic()
        )
    end

    ### Constructor from config
    function StochasticInterpolant(
        config::Config.Hyperparameters,
        trait = Models.Stochastic()
    )

        # Define velocity model
        velocity_model = Architectures.DenseNeuralNetwork(
            config.architecture.in_features,
            config.architecture.out_features,
            config.architecture.hidden_features;
        );
        if trait == Models.Deterministic()
            return StochasticInterpolant(config.model.interpolant_type, velocity_model)

        elseif trait == Models.Stochastic()
            # Define score model
            score_model = Architectures.DenseNeuralNetwork(
                config.architecture.in_features,
                config.architecture.out_features,
                config.architecture.hidden_features;
            );
        end

        return StochasticInterpolant(
            config.model.interpolant_type,
            velocity_model,
            score_model
        )
    end
end

"""
    drift_term(
        ::Models.Stochastic,
        model::StochasticInterpolant, 
        diffusion_fn, 
        x, 
        ps, 
        st
    )

    Compute the drift term for a stochastic interpolant.
"""
function drift_term(model::StochasticInterpolant, diffusion_fn)
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
function drift_term(::Models.Stochastic, model::StochasticInterpolant, diffusion_fn)
    function drift_wrapper(x, ps, st; model = model)
        velocity, _velocity_st = model.velocity(x, ps.velocity, st.velocity)
        score, _score_st = model.score(x, ps.score, st.score)
        st = (; velocity = _velocity_st, score = _score_st)

        _, t = x
        diffusion = diffusion_fn(t)
        return velocity .+ 0.5f0 .* diffusion .^ 2 .* score, st
    end

    return drift_wrapper
end
