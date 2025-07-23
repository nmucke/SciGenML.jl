"""
    load_and_reshape_data(
        num_steps,
        num_skip_steps,
        i
    )

Loads the KNMI data for the given number of steps.
"""
function load_and_reshape_data(num_steps, num_skip_steps, i)
    data = NPZ.npzread("data/knmi/sim_$i.npz")
    tas = data["tas"]
    ym = data["ym"]

    tas = tas[1:num_skip_steps:(num_steps * num_skip_steps), :, :]
    ym = ym[1:num_skip_steps:(num_steps * num_skip_steps), :, :]

    tas = permutedims(tas, (2, 3, 1))
    ym = permutedims(ym, (2, 3, 1))

    tas = reshape(tas, (64, 128, 1, num_steps))
    ym = reshape(ym, (64, 128, 1, num_steps))

    return tas, ym
end

"""
    load_knmi_data(
        num_steps
    )

    Loads the KNMI data for the given number of steps.
"""
function load_knmi_tranining_data(num_steps, trajectories, num_skip_steps)
    base = []
    target = []
    field_conditioning = []
    for i in trajectories
        tas, ym = load_and_reshape_data(num_steps, num_skip_steps, i)

        push!(base, tas[:, :, :, 1:(end - 1)])
        push!(target, tas[:, :, :, 2:end])
        push!(
            field_conditioning,
            cat(tas[:, :, :, 1:(end - 1)], ym[:, :, :, 1:(end - 1)], dims = 3)
        )
    end

    return (;
        base = cat(base..., dims = 4),
        target = cat(target..., dims = 4),
        field_conditioning = cat(field_conditioning..., dims = 4)
    )
end

"""
    load_knmi_test_data(
        num_steps,
        trajectories,
        num_skip_steps
    )

Loads the KNMI test data for the given number of steps.
"""
function load_knmi_test_data(num_steps, trajectories, num_skip_steps)
    base = []
    field_conditioning = []

    for i in trajectories
        tas, ym = load_and_reshape_data(num_steps, num_skip_steps, i)

        push!(base, tas)
        push!(field_conditioning, cat(tas, ym, dims = 3))
    end

    return (; base = stack(base), field_conditioning = stack(field_conditioning))
end

"""
    load_knmi_data(
        config::Config.KNMIDataHyperparameters;
        kwargs...
    )

Loads the KNMI data for the given configuration.
"""
function load_knmi_data(config::Config.KNMIDataHyperparameters; kwargs...)
    println("Loading KNMI training data...")
    train_data = load_knmi_tranining_data(
        config.train_num_steps,
        config.train_trajectories,
        config.num_skip_steps
    )

    println("Loading KNMI test data...")
    test_data = load_knmi_test_data(
        config.test_num_steps,
        config.test_trajectories,
        config.num_skip_steps
    )

    return train_data, test_data
end
