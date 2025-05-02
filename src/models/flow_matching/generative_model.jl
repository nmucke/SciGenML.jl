"""
    FlowMatching

A flow matching generative model.

The flow matching generative model is a generative model that uses a flow matching approach to generate data.

It is a special case of the stochastic interpolant generative model where there is no
noise in the interpolant.
"""
mutable struct FlowMatching <: Models.GenerativeModel
    interpolant_coefs::Any
    velocity::Any
    ps::Any
    st::Any
    trait::Any

    function FlowMatching(velocity,)
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)

        ps = (; velocity = velocity_ps)
        st = (; velocity = velocity_st)
        return new(
            linear_interpolant_coefs(Models.Deterministic()),
            velocity,
            ps,
            st,
            Models.Deterministic()
        )
    end

    # Constructor with interpolant type
    function FlowMatching(interpolant_type::String, velocity)
        velocity_ps, velocity_st = Lux.setup(Lux.Random.default_rng(), velocity)

        ps = (; velocity = velocity_ps)
        st = (; velocity = velocity_st)
        return new(
            get_interpolant_coefs(Models.Deterministic(), interpolant_type),
            velocity,
            ps,
            st,
            Models.Deterministic()
        )
    end

    ### Constructor from config
    function FlowMatching(config::Config.Hyperparameters,)

        # Define velocity model
        velocity_model = Architectures.DenseNeuralNetwork(
            config.architecture.in_features,
            config.architecture.out_features,
            config.architecture.hidden_features;
        );
        return FlowMatching(config.model.interpolant_type, velocity_model)
    end
end
