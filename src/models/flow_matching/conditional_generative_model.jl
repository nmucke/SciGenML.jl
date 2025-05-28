"""
    ConditionalFlowMatching

A conditional flow matching generative model.

The conditional flow matching generative model is a generative model that uses a flow matching approach to generate data.

It is a special case of the conditional stochastic interpolant generative model where there is no
noise in the interpolant.
"""
mutable struct ConditionalFlowMatching <: Models.ConditionalGenerativeModel
    interpolant_coefs::Any
    velocity::Any
    ps::Any
    st::Any
    trait::Any
    device::Any
    guidance_scale::DEFAULT_TYPE
    replacement_probability::DEFAULT_TYPE
    unconditional_condition::Any

    function ConditionalFlowMatching(
        velocity;
        guidance_scale = 0.5f0,
        replacement_probability = 0.25f0,
        unconditional_condition = 10.0f0
    )
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)

        ps = (; velocity = velocity_ps)
        st = (; velocity = velocity_st)
        return new(
            linear_interpolant_coefs(Models.Deterministic()),
            velocity,
            ps,
            st,
            Models.Deterministic(),
            DEFAULT_DEVICE,
            guidance_scale,
            replacement_probability,
            unconditional_condition
        )
    end

    # Constructor with interpolant type
    function ConditionalFlowMatching(
        interpolant_type::String,
        velocity;
        guidance_scale = 0.5f0,
        replacement_probability = 0.25f0,
        unconditional_condition = 10.0f0
    )
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)

        ps = (; velocity = velocity_ps)
        st = (; velocity = velocity_st)
        return new(
            get_interpolant_coefs(Models.Deterministic(), interpolant_type),
            velocity,
            ps,
            st,
            Models.Deterministic(),
            DEFAULT_DEVICE,
            guidance_scale,
            replacement_probability,
            unconditional_condition
        )
    end

    ### Constructor from config
    function ConditionalFlowMatching(config::Config.Hyperparameters,)

        # Define velocity model
        velocity_model = Architectures.get_architecture(config.architecture);

        return ConditionalFlowMatching(
            config.model.interpolant_type,
            velocity_model,
            guidance_scale = config.model.guidance_scale,
            replacement_probability = config.model.replacement_probability,
            unconditional_condition = config.model.unconditional_condition
        )
    end
end

"""
    drift_term(
        ::Models.Stochastic,
        model::Models.ConditionalFlowMatching,
    )
    
"""
function drift_term(model::Models.ConditionalFlowMatching)
    return drift_term(model.trait, model)
end

"""
    drift_term(
        ::Models.Deterministic,
        model::Models.ConditionalFlowMatching,
    )

    Get the drift term for the conditional flow matching generative model.
"""
function drift_term(::Models.Deterministic, model::Models.ConditionalFlowMatching)
    function drift_wrapper(x, ps, st; model = model)
        num_samples = size(x[1])[end]
        unconditional_condition =
            model.unconditional_condition .* ones(DEFAULT_TYPE, 1, num_samples)
        unconditional_condition = unconditional_condition |> model.device

        x_unconditional = (x[1], unconditional_condition, x[3])

        unconditional_velocity, _st =
            model.velocity(x_unconditional, ps.velocity, st.velocity)
        st = (; velocity = _st)
        conditional_velocity, _st = model.velocity(x, ps.velocity, st.velocity)
        st = (; velocity = _st)

        return model.guidance_scale .* conditional_velocity .+
               (1.0f0 - model.guidance_scale) .* unconditional_velocity,
        st
    end

    return drift_wrapper
end
