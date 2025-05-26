
"""
    sde_integrator(
        stepper,
        drift_term_fn,
        diffusion_term_fn,
        x,
        num_steps,
        ps,
        st;
        t_interval = [0.0f0, 1.0f0],
        verbose::Bool = true,
        rng::Random.AbstractRNG = Lux.Random.default_rng()
    )

    Integrate SDE using a given stepper.
"""

function sde_integrator(
    stepper,
    drift_term_fn,
    diffusion_term_fn,
    x,
    num_steps,
    ps,
    st;
    device = DEFAULT_DEVICE,
    t_interval = [0.0f0, 1.0f0],
    verbose::Bool = true,
    rng::Random.AbstractRNG = Lux.Random.default_rng()
)
    dt = (t_interval[2] - t_interval[1]) / num_steps |> DEFAULT_TYPE |> device
    t = t_interval[1] .* ones(DEFAULT_TYPE, (1, size(x)[end])) |> device

    iter = Utils.get_iter(num_steps, verbose)
    for i in iter
        x, t, st = stepper(drift_term_fn, diffusion_term_fn, x, t, dt, ps, st; rng = rng)
    end

    return x, st
end

"""
    sde_integrator(
        stepper,
        drift_term_fn,
        diffusion_term_fn,
        x,
        scalar_conditioning,
        num_steps,
        ps,
        st;
        t_interval = [0.0f0, 1.0f0],
        verbose::Bool = true,
        rng::Random.AbstractRNG = Lux.Random.default_rng()
    )

    Integrate SDE using a given stepper.
"""

function sde_integrator(
    stepper,
    drift_term_fn,
    diffusion_term_fn,
    x,
    scalar_conditioning,
    num_steps,
    ps,
    st;
    device = DEFAULT_DEVICE,
    t_interval = [0.0f0, 1.0f0],
    verbose::Bool = true,
    rng::Random.AbstractRNG = Lux.Random.default_rng()
)
    dt = (t_interval[2] - t_interval[1]) / num_steps |> DEFAULT_TYPE |> device
    t = t_interval[1] .* ones(DEFAULT_TYPE, (1, size(x)[end])) |> device

    iter = Utils.get_iter(num_steps, verbose)
    for i in iter
        x, t, st = stepper(
            drift_term_fn,
            diffusion_term_fn,
            x,
            scalar_conditioning,
            t,
            dt,
            ps,
            st;
            rng = rng
        )
    end

    return x, st
end

"""
    Euler Maruyama step for SDEs.
"""
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
    z = Random.randn!(rng, similar(x, size(x)))

    # Get diffusion
    diffusion = diffusion_term_fn(t)
    diffusion = reshape(
        diffusion,
        ntuple(i -> i == ndims(x[1]) ? size(diffusion)[end] : 1, ndims(x[1]))
    )
    x = x .+ sqrt(dt) .* diffusion .* z

    # Update time
    t = t .+ dt
    return x, t, st
end

"""
    Euler Maruyama step for SDEs.
"""
function euler_maruyama_step(
    drift_term_fn,
    diffusion_term_fn,
    x,
    scalar_conditioning,
    t,
    dt,
    ps,
    st;
    rng = Lux.Random.default_rng()
)

    # Get drift
    drift, st = drift_term_fn((x, scalar_conditioning, t), ps, st)
    x = x .+ dt .* drift

    # Get gaussian noise
    z = Random.randn!(rng, similar(x, size(x)))

    # Get diffusion
    diffusion = diffusion_term_fn(t)
    diffusion = reshape(
        diffusion,
        ntuple(i -> i == ndims(x[1]) ? size(diffusion)[end] : 1, ndims(x[1]))
    )
    x = x .+ sqrt(dt) .* diffusion .* z

    # Update time
    t = t .+ dt
    return x, t, st
end

"""
    Heun step for SDEs.
"""
function heun_step(
    drift_term_fn,
    diffusion_term_fn,
    x,
    t,
    dt,
    ps,
    st;
    rng = Lux.Random.default_rng()
)
    ### Euler Maruyama step ###
    # Get drift
    em_drift, st = drift_term_fn((x, t), ps, st)
    em_x = x .+ dt .* em_drift

    # Get gaussian noise
    z = Random.randn!(rng, similar(x, size(x)))

    # Get diffusion
    em_diffusion = diffusion_term_fn(t)
    em_diffusion = reshape(
        em_diffusion,
        ntuple(i -> i == ndims(x) ? size(em_diffusion)[end] : 1, ndims(x))
    )
    em_x = em_x .+ sqrt(dt) .* em_diffusion .* z

    ### Corrector step ###
    # Get drift
    corrected_drift, st = drift_term_fn((em_x, t .+ dt), ps, st)

    # Get diffusion
    corrected_diffusion = diffusion_term_fn(t .+ dt)
    corrected_diffusion = reshape(
        corrected_diffusion,
        ntuple(i -> i == ndims(x) ? size(corrected_diffusion)[end] : 1, ndims(x))
    )

    # Get final drift and diffusion
    final_drift = (em_drift .+ corrected_drift) / 2
    x = x .+ dt .* final_drift

    # Get final diffusion
    final_diffusion = (em_diffusion .+ corrected_diffusion) / 2
    x = x .+ sqrt(dt) .* final_diffusion .* z

    # Update time
    t = t .+ dt
    return x, t, st
end

"""
    Heun step for SDEs.
"""
function heun_step(
    drift_term_fn,
    diffusion_term_fn,
    x,
    scalar_conditioning,
    t,
    dt,
    ps,
    st;
    rng = Lux.Random.default_rng()
)
    ### Euler Maruyama step ###
    # Get drift
    em_drift, st = drift_term_fn((x, scalar_conditioning, t), ps, st)
    em_x = x .+ dt .* em_drift

    # Get gaussian noise
    z = Random.randn!(rng, similar(x, size(x)))

    # Get diffusion
    em_diffusion = diffusion_term_fn(t)
    em_diffusion = reshape(
        em_diffusion,
        ntuple(i -> i == ndims(x) ? size(em_diffusion)[end] : 1, ndims(x))
    )
    em_x = em_x .+ sqrt(dt) .* em_diffusion .* z

    ### Corrector step ###
    # Get drift
    corrected_drift, st = drift_term_fn((em_x, scalar_conditioning, t .+ dt), ps, st)

    # Get diffusion
    corrected_diffusion = diffusion_term_fn(t .+ dt)
    corrected_diffusion = reshape(
        corrected_diffusion,
        ntuple(i -> i == ndims(x) ? size(corrected_diffusion)[end] : 1, ndims(x))
    )

    # Get final drift and diffusion
    final_drift = (em_drift .+ corrected_drift) / 2
    x = x .+ dt .* final_drift

    # Get final diffusion
    final_diffusion = (em_diffusion .+ corrected_diffusion) / 2
    x = x .+ sqrt(dt) .* final_diffusion .* z

    # Update time
    t = t .+ dt
    return x, t, st
end
