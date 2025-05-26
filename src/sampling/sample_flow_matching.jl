
"""
    sample(
        ::Models.Deterministic,
        model::Models.FlowMatching,
        num_samples::Int,
        num_steps::Int,
        rng::Random.AbstractRNG = Random.default_rng()
    )

    Sample from a stochastic interpolant generative model using the forward Euler method.
"""
function sample(
    ::Models.Deterministic,
    model::Models.FlowMatching,
    num_steps::Int;
    num_samples::Int = 1000,
    prior_samples = nothing,
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = true,
    stepper = TimeIntegrators.RK4_step
)
    model.st = (; velocity = Lux.testmode(model.st.velocity))

    if isnothing(prior_samples)
        x_samples =
            rand(rng, Distributions.Normal(0.0, 1.0), (1, num_samples)) .|>
            DEFAULT_TYPE |>
            model.device
    else
        x_samples = prior_samples |> model.device
        num_samples = size(x_samples)[end]
    end

    x_samples, velocity_st = TimeIntegrators.ode_integrator(
        stepper,
        model.velocity,
        x_samples,
        num_steps,
        model.ps.velocity,
        model.st.velocity;
        t_interval = [0.0f0, 1.0f0],
        verbose = verbose,
        device = model.device
    )
    st = (; velocity = velocity_st)

    return x_samples, st
end

"""
    sample(
        ::Models.Deterministic,
        model::Models.FlowMatching,
        num_samples::Int,
        num_steps::Int,
        rng::Random.AbstractRNG = Random.default_rng()
    )

    Sample from a stochastic interpolant generative model using the forward Euler method.
"""
function sample(
    ::Models.Deterministic,
    model::Models.FlowMatching,
    scalar_conditioning,
    num_steps::Int;
    num_samples::Int = 1000,
    prior_samples = nothing,
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = true,
    stepper = TimeIntegrators.RK4_step
)
    model.st = (; velocity = Lux.testmode(model.st.velocity))

    scalar_conditioning = scalar_conditioning |> model.device

    if isnothing(prior_samples)
        x_samples =
            rand(rng, Distributions.Normal(0.0, 1.0), (1, num_samples)) .|>
            DEFAULT_TYPE |>
            model.device
    else
        x_samples = prior_samples |> model.device
        num_samples = size(x_samples)[end]
    end

    x_samples, velocity_st = TimeIntegrators.ode_integrator(
        stepper,
        model.velocity,
        x_samples,
        scalar_conditioning,
        num_steps,
        model.ps.velocity,
        model.st.velocity;
        t_interval = [0.0f0, 1.0f0],
        verbose = verbose,
        device = model.device
    )
    st = (; velocity = velocity_st)

    return x_samples, st
end
