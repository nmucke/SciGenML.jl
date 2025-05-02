
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
    verbose::Bool = true
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
    t_i = zeros(DEFAULT_TYPE, (1, num_samples))

    iter = Utils.get_iter(num_steps, verbose)

    for i in iter
        velocity, _velocity_st =
            model.velocity((x_samples, t_i), model.ps.velocity, model.st.velocity)
        score, _score_st = model.score((x_samples, t_i), model.ps.score, model.st.score)
        model.st = (; velocity = _velocity_st, score = _score_st)

        z_samples = Random.randn!(rng, similar(x_samples, (1, num_samples)))
        diffusion = diffusion_fn(t_i)

        diffusion_term = diffusion .* z_samples
        drift_term = velocity .+ 0.5f0 .* diffusion .^ 2 .* score

        x_samples = x_samples .+ dt .* drift_term + sqrt.(dt) .* diffusion_term
        t_i = t_i .+ dt
    end

    velocity, _velocity_st =
        model.velocity((x_samples, t_i), model.ps.velocity, model.st.velocity)
    model.st = (; velocity = _velocity_st, score = model.st.score)

    x_samples = x_samples .+ dt .* velocity

    return x_samples
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

    dt = 1.0 / num_steps |> DEFAULT_TYPE

    t_i = zeros(DEFAULT_TYPE, (1, num_samples))

    iter = Utils.get_iter(num_steps, verbose)

    x_samples, st = TimeIntegrators.ode_integrator(
        stepper,
        model.velocity,
        x_samples,
        num_steps,
        model.ps.velocity,
        model.st.velocity;
        t_interval = [0.0, 1.0],
        verbose = verbose
    )

    # for i in iter

    #     x_samples, velocity_st = integrator(
    #         model.velocity,
    #         x_samples,
    #         t_i,
    #         dt,
    #         model.ps.velocity,
    #         model.st.velocity
    #     )

    #     model.st =(; 
    #         velocity = velocity_st, 
    #         score = model.trait == Models.Stochastic() ? model.st.score : nothing
    #     )

    #     t_i = t_i .+ dt
    # end

    return x_samples
end
