
"""
    load_knmi_data(
        num_steps
    )

    Loads the KNMI data for the given number of steps.
"""
function load_knmi_data(num_steps, num_trajectories, num_skip_steps)
    println("Loading KNMI data...")

    base = []
    target = []
    field_conditioning = []
    for i in 1:num_trajectories
        data = NPZ.npzread("data/knmi/sim_$i.npz")

        tas = data["tas"]
        ym = data["ym"]

        tas = (tas .- Statistics.mean(tas)) ./ Statistics.std(tas)
        ym = (ym .- Statistics.mean(ym)) ./ Statistics.std(ym)

        tas = tas[1:num_skip_steps:(num_steps * num_skip_steps), :, :]
        ym = ym[1:num_skip_steps:(num_steps * num_skip_steps), :, :]

        tas = permutedims(tas, (2, 3, 1))
        ym = permutedims(ym, (2, 3, 1))

        tas = reshape(tas, (64, 128, 1, num_steps))
        ym = reshape(ym, (64, 128, 1, num_steps))

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

function load_knmi_data(config::Config.KNMIDataHyperparameters; kwargs...)
    return load_knmi_data(config.num_steps, config.num_trajectories, config.num_skip_steps)
end
