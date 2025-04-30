"""
    StochasticInterpolantGenerativeModel

    A generative model that uses a stochastic interpolant to generate data.
"""
struct StochasticInterpolantGenerativeModel <: GenerativeModel
    interpolant_coefs::InterpolantCoefs
    drift_model::Any
    ps::NamedTuple
    st::NamedTuple

    # Constructor
    function StochasticInterpolantGenerativeModel(drift_model, ps, st)
        return new(linear_interpolant_coefs(), drift_model, ps, st)
    end

    function StochasticInterpolantGenerativeModel(
        interpolant_type::String,
        drift_model,
        ps,
        st
    )
        return new(get_interpolant_coefs(interpolant_type), drift_model, ps, st)
    end

    function StochasticInterpolantGenerativeModel(interpolant_type::String, drift_model)
        ps, st = Lux.setup(Lux.Random.default_rng(), drift_model)
        return new(get_interpolant_coefs(interpolant_type), drift_model, ps, st)
    end
end
