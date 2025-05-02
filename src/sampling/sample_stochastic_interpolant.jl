
"""
    sample(
        ::Models.Stochastic,
        model::Models.StochasticInterpolantGenerativeModel,
        num_samples::Int,
        num_steps::Int,
        rng::Random.AbstractRNG = Random.default_rng()
    )

    Sample from a stochastic interpolant generative model using the forward Euler method.
"""
function sample(
    ::Models.Stochastic,
    model::Models.StochasticInterpolant,
    num_steps::Int;
    num_samples::Int = 1000,
    prior_samples = nothing,
    diffusion_fn = nothing,
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = true,
    stepper = TimeIntegrators.heun_step
)
    model.st =
        (; velocity = Lux.testmode(model.st.velocity), score = Lux.testmode(model.st.score))

    if isnothing(diffusion_fn)
        diffusion_fn = t -> model.interpolant_coefs.gamma(t)
    end

    if isnothing(prior_samples)
        x_samples =
            rand(rng, Distributions.Normal(0.0, 1.0), (1, num_samples)) .|> DEFAULT_TYPE
    else
        x_samples = prior_samples
        num_samples = size(x_samples)[end]
    end

    dt = 1.0 / num_steps |> DEFAULT_TYPE

    # Get drift term
    drift_term_fn = Models.drift_term(model, diffusion_fn)

    # Solve SDE
    x_samples, st = TimeIntegrators.sde_integrator(
        stepper,
        drift_term_fn,
        diffusion_fn,
        x_samples,
        num_steps,
        model.ps,
        model.st;
        t_interval = [0.0f0, 1.0f0 - dt],
        verbose = verbose,
        rng = rng
    )

    # Compute last step using forward Euler
    t_end = (1.0f0 - dt) .* ones(DEFAULT_TYPE, (1, num_samples))
    x_samples, _, velocity_st = TimeIntegrators.RK4_step(
        model.velocity,
        x_samples,
        t_end,
        dt,
        model.ps.velocity,
        model.st.velocity
    )
    st = (; velocity = velocity_st, score = model.st.score)

    return x_samples, st
end

"""
    sample(
        ::Models.Deterministic,
        model::Models.StochasticInterpolantGenerativeModel,
        num_samples::Int,
        num_steps::Int,
        rng::Random.AbstractRNG = Random.default_rng()
    )

    Sample from a stochastic interpolant generative model using the forward Euler method.
"""
function sample(
    ::Models.Deterministic,
    model::Models.StochasticInterpolant,
    num_steps::Int;
    num_samples::Int = 1000,
    prior_samples = nothing,
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = true,
    stepper = TimeIntegrators.RK4_step
)
    model.st = (;
        velocity = Lux.testmode(model.st.velocity),
        score = model.trait == Models.Stochastic() ? Lux.testmode(model.st.score) : nothing
    )

    if isnothing(prior_samples)
        x_samples =
            rand(rng, Distributions.Normal(0.0, 1.0), (1, num_samples)) .|> DEFAULT_TYPE
    else
        x_samples = prior_samples
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
        verbose = verbose
    )
    st = (; velocity = velocity_st, score = model.st.score)

    return x_samples, st
end
