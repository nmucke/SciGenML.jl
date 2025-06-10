
"""
    sample(
        ::Models.Stochastic,
        model::Models.FollmerStochasticInterpolant,
        scalar_conditioning,
        num_steps::Int,
        num_samples::Int,
        prior_samples = nothing,
        diffusion_fn = nothing,
        rng::Random.AbstractRNG = Random.default_rng(),
        verbose::Bool = true,
        stepper = TimeIntegrators.heun_step
    )

    Sample from a stochastic interpolant generative model using the forward Euler method.
"""

function posterior_sample(
    model::Models.FollmerStochasticInterpolant,
    conditioning,
    observations,
    observation_operator,
    log_likelihood_fn::Function,
    num_steps::Int;
    num_samples::Int = 1000,
    prior_samples = nothing,
    diffusion_fn = nothing,
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = true,
    stepper = TimeIntegrators.heun_step,
    ode_stepper = TimeIntegrators.RK4_step
)
    model.st = (; velocity = Lux.testmode(model.st.velocity))

    if !(typeof(conditioning) <: Tuple)
        conditioning = (conditioning,)
    end
    conditioning = conditioning .|> model.device

    if isnothing(prior_samples)
        x_samples =
            rand(rng, Distributions.Normal(0.0, 1.0), (1, num_samples)) .|>
            DEFAULT_TYPE |>
            model.device
    else
        x_samples = prior_samples |> model.device
        num_samples = size(x_samples)[end]
    end

    dt = 1.0 / num_steps |> DEFAULT_TYPE

    if isnothing(diffusion_fn)
        diffusion_fn = t -> model.interpolant_coefs.gamma(t)
    end

    # Solve SDE
    x_samples, st = TimeIntegrators.sde_integrator(
        stepper,
        Models.posterior_drift_term(
            model,
            observation_operator,
            log_likelihood_fn,
            diffusion_fn
        ),
        diffusion_fn,
        x_samples,
        (conditioning..., observations),
        num_steps,
        model.ps,
        model.st;
        t_interval = [0.0f0, 1.0f0 - dt],
        verbose = verbose,
        rng = rng,
        device = model.device
    )
    # model.st = (; velocity = st.velocity)

    x_samples, st = TimeIntegrators.ode_integrator(
        ode_stepper,
        Models.posterior_drift_term(
            model,
            observation_operator,
            log_likelihood_fn,
            diffusion_fn
        ),
        x_samples,
        (conditioning..., observations),
        1,
        model.ps,
        model.st;
        t_interval = [1.0f0 - dt, 1.0f0],
        verbose = false,
        device = model.device
    )

    # st = (; velocity = st.velocity)

    return x_samples, st
end
