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
    function FollmerStochasticInterpolant(config::Config.Hy
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
            endperparameters,)

        # Define velocity model
        velocity_model = Architectures.get_architecture(config.architecture);

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

function compute_score(model::FollmerStochasticInterpolant, velocity, x)
    x, f, t = x
    t = Utils.reshape_scalar(t, ndims(x))

    alpha = model.interpolant_coefs.alpha(t)
    beta = model.interpolant_coefs.beta(t)

    alpha_diff = model.interpolant_coefs.alpha_diff(t)
    beta_diff = model.interpolant_coefs.beta_diff(t)

    gamma = model.interpolant_coefs.gamma(t)
    gamma_diff = model.interpolant_coefs.gamma_diff(t)

    A = t .* gamma .* (beta_diff .* gamma .- gamma_diff .* beta)
    A = 1.0f0 ./ (A .+ ZERO_TOL)
    c = beta_diff .* x .+ (beta .* alpha_diff - beta_diff .* alpha) .* f

    return A .* (beta .* velocity .- c)
end

"""
    drift_term(
        model::FollmerGenerativeModel, 
    )

    Compute the drift term for a stochsatic interpolant.
"""
function drift_term(model::FollmerStochasticInterpolant, diffusion_fn = nothing)
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
function drift_term(model::FollmerStochasticInterpolant, diffusion_fn::Function)
    function drift_wrapper(x, ps, st; model = model)
        velocity, _velocity_st = model.velocity(x, ps.velocity, st.velocity)
        st = (; velocity = _velocity_st)

        score = compute_score(model, velocity, x)

        t = x[end]
        t = Utils.reshape_scalar(t, ndims(x[1]))

        diffusion = diffusion_fn(t)

        gamma = model.interpolant_coefs.gamma(t)

        return velocity + 0.5f0 .* (diffusion .^ 2 - gamma .^ 2) .* score, st
    end

    return drift_wrapper
end
