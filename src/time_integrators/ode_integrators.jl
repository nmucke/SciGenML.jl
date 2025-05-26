"""
    ode_integrator(
        stepper, 
        rhs, 
        x, 
        num_steps, 
        ps, 
        st;
        t_interval=[0.0, 1.0], 
        verbose::Bool = true
    )

    Generic ODE integrator.
"""
function ode_integrator(
    stepper,
    rhs,
    x,
    num_steps,
    ps,
    st;
    device = DEFAULT_DEVICE,
    t_interval = [0.0, 1.0],
    verbose::Bool = true
)
    dt = (t_interval[2] - t_interval[1]) / num_steps |> DEFAULT_TYPE |> device
    t = t_interval[1] .* ones(DEFAULT_TYPE, (1, size(x)[end])) |> device

    iter = Utils.get_iter(num_steps, verbose)
    for i in iter
        x, t, st = stepper(rhs, x, t, dt, ps, st)
    end
    return x, st
end

"""
    ode_integrator(
        stepper, 
        rhs, 
        x, 
        scalar_conditioning,
        num_steps, 
        ps, 
        st;
        t_interval=[0.0, 1.0], 
        verbose::Bool = true
    )

    Generic ODE integrator.
"""
function ode_integrator(
    stepper,
    rhs,
    x,
    scalar_conditioning,
    num_steps,
    ps,
    st;
    device = DEFAULT_DEVICE,
    t_interval = [0.0, 1.0],
    verbose::Bool = true
)
    dt = (t_interval[2] - t_interval[1]) / num_steps |> DEFAULT_TYPE |> device
    t = t_interval[1] .* ones(DEFAULT_TYPE, (1, size(x)[end])) |> device

    iter = Utils.get_iter(num_steps, verbose)
    for i in iter
        x, t, st = stepper(rhs, x, scalar_conditioning, t, dt, ps, st)
    end
    return x, st
end

"""
    forward_euler_step(model, x, t, dt)

    Perform a forward Euler step.
"""
function forward_euler_step(rhs, x, t, dt, ps, st)

    # Get rhs_output
    rhs_output, st = rhs((x, t), ps, st)

    # Compute next state
    x_next = x .+ dt .* rhs_output

    # Update time
    t = t .+ dt

    return x_next, t, st
end

"""
    forward_euler_step(model, x, scalar_conditioning, t, dt)

    Perform a forward Euler step.
"""
function forward_euler_step(rhs, x, scalar_conditioning, t, dt, ps, st)

    # Get rhs_output
    rhs_output, st = rhs((x, scalar_conditioning, t), ps, st)

    # Compute next state
    x_next = x .+ dt .* rhs_output

    # Update time
    t = t .+ dt

    return x_next, t, st
end

"""
    RK4_step(rhs, x, t, dt, ps, st)

    Perform a 4th order Runge-Kutta step.
"""
function RK4_step(rhs, x, t, dt, ps, st)

    # Get rhs_output
    stage_1, st = rhs((x, t), ps, st)
    stage_2, st = rhs((x .+ dt * stage_1 ./ 2.0f0, t .+ dt ./ 2.0f0), ps, st)
    stage_3, st = rhs((x .+ dt * stage_2 ./ 2.0f0, t .+ dt ./ 2.0f0), ps, st)
    stage_4, st = rhs((x .+ dt * stage_3, t .+ dt), ps, st)

    # Compute next state
    x_next = x .+ dt .* (stage_1 .+ 2.0f0 * stage_2 .+ 2.0f0 * stage_3 .+ stage_4) ./ 6.0f0

    # Update time
    t = t .+ dt

    return x_next, t, st
end

"""
    RK4_step(rhs, x, scalar_conditioning, t, dt, ps, st)

    Perform a 4th order Runge-Kutta step.
"""
function RK4_step(rhs, x, scalar_conditioning, t, dt, ps, st)

    # Get rhs_output
    stage_1, st = rhs((x, scalar_conditioning, t), ps, st)
    stage_2, st =
        rhs((x .+ dt * stage_1 ./ 2.0f0, scalar_conditioning, t .+ dt ./ 2.0f0), ps, st)
    stage_3, st =
        rhs((x .+ dt * stage_2 ./ 2.0f0, scalar_conditioning, t .+ dt ./ 2.0f0), ps, st)
    stage_4, st = rhs((x .+ dt * stage_3, scalar_conditioning, t .+ dt), ps, st)

    # Compute next state
    x_next = x .+ dt .* (stage_1 .+ 2.0f0 * stage_2 .+ 2.0f0 * stage_3 .+ stage_4) ./ 6.0f0

    # Update time
    t = t .+ dt

    return x_next, t, st
end
