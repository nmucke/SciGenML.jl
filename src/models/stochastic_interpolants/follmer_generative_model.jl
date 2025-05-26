"""
    FollmerStochasticInterpolant

A follmer stochastic interpolant generative model.
"""
mutable struct FollmerStochasticInterpolant <: Models.ConditionalGenerativeModel
    interpolant_coefs::Any
    velocity::Any
    ps::Any
    st::Any
    trait::Any
    device::Any

    # Constructor with velocity
    function FollmerStochasticInterpolant(velocity)
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)

        ps = (; velocity = velocity_ps)
        st = (; velocity = velocity_st)
        return new(
            quadratic_interpolant_coefs(Models.Stochastic()),
            velocity,
            ps,
            st,
            Models.Stochastic(),
            DEFAULT_DEVICE
        )
    end

    # Constructor with interpolant type
    function FollmerStochasticInterpolant(interpolant_type::String, velocity)
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)

        ps = (; velocity = velocity_ps)
        st = (; velocity = velocity_st)
        return new(
            get_interpolant_coefs(Models.Stochastic(), interpolant_type),
            velocity,
            ps,
            st,
            Models.Stochastic(),
            DEFAULT_DEVICE
        )
    end

    ### Constructor from config
    function FollmerStochasticInterpolant(config::Config.Hyperparameters,)

        # Define velocity model
        velocity_model = Architectures.DenseNeuralNetwork(
            config.architecture.in_features,
            config.architecture.out_features,
            config.architecture.hidden_features;
        );

        return FollmerStochasticInterpolant(config.model.interpolant_type, velocity_model)
    end

    function FollmerStochasticInterpolant(
        interpolant_coefs::Any,
        velocity::Any,
        ps::Any,
        st::Any,
        trait::Any
    )
        return new(interpolant_coefs, velocity, ps, st, trait, device)
    end
end

function compute_score(model::FollmerStochasticInterpolant, x, ps, st, diffusion_fn)
    velocity, _velocity_st = model.velocity(x, ps.velocity, st.velocity)
    st = (; velocity = _velocity_st)

    return velocity, st
end

"""
    drift_term(
        model::FollmerGenerativeModel, 
    )

    Compute the drift term for a stochsatic interpolant.
"""
function drift_term(model::FollmerStochasticInterpolant)
    function drift_wrapper(x, ps, st; model = model)
        velocity, _velocity_st = model.velocity(x, ps.velocity, st.velocity)
        st = (; velocity = _velocity_st)

        return velocity, st
    end

    return drift_wrapper
end

"""
    drift_term(
        model::FollmerGenerativeModel, 
        diffusion_fn
    )

    Compute the drift term for a stochsatic interpolant.
"""
function drift_term(model::FollmerStochasticInterpolant, diffusion_fn)
    function drift_wrapper(x, ps, st; model = model)
        velocity, _velocity_st = model.velocity(x, ps.velocity, st.velocity)
        st = (; velocity = _velocity_st)

        _, t = x
        diffusion = diffusion_fn(t)
        diffusion = reshape(
            diffusion,
            ntuple(i -> i == ndims(x[1]) ? size(diffusion)[end] : 1, ndims(x[1]))
        )
        return velocity
    end
    return drift_wrapper
end
