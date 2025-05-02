
function sde_integrator(
    stepper,
    drift_term_fn,
    diffusion_term_fn,
    x,
    num_steps,
    ps,
    st;
    t_interval = [0.0, 1.0],
    verbose::Bool = true,
    rng::Random.AbstractRNG = Lux.Random.default_rng()
)
    dt = (t_interval[2] - t_interval[1]) / num_steps
    t = t_interval[1] .* ones(DEFAULT_TYPE, (1, size(x)[end]))

    iter = Utils.get_iter(num_steps, verbose)
    for i in iter
        x, t, st = stepper(drift_term_fn, diffusion_term_fn, x, t, dt, ps, st; rng = rng)
    end

    return x, st
end

function euler_maruyama_step(
    drift_term_fn,
    diffusion_term_fn,
    x,
    t,
    dt,
    ps,
    st;
    rng = Lux.Random.default_rng()
)

    # Get drift
    drift, st = drift_term_fn((x, t), ps, st)
    x = x .+ dt .* drift

    # Get gaussian noise
    z = Random.randn!(rng, similar(x, (1, size(x)[end])))

    # Get diffusion
    diffusion = diffusion_term_fn(t)
    x = x .+ sqrt(dt) .* diffusion .* z

    # Update time
    t = t .+ dt
    return x, t, st
end
