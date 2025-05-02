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
end
