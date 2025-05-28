"""
    load_kolmogorov_data(;
        num_trajectories = 5,
        num_skip_steps = 5,
        num_steps = 100,
        start_step = 500
    )
    Load the Kolmogorov data.

    Outputs a named tuple with the following fields:
    - `base`: The base data. [H, W, C, (T-1)*N]
    - `target`: The target data. [H, W, C, (T-1)*N]
    - `field_conditioning`: The field conditioning data. [H, W, C, (T-1)*N]
"""
function load_kolmogorov_data(
    num_trajectories = 5,
    num_skip_steps = 5,
    num_steps = 100,
    start_step = 500,
    normal_base_distribution = false
)
    data = []
    for i in 1:num_trajectories
        u = JLD2.load("data/kolmogorov_128/sim_$(i).jld2")
        u = u["u"][
            :,
            :,
            :,
            start_step:num_skip_steps:(start_step + num_skip_steps * num_steps - 1)
        ]
        u = u[2:(end - 1), 2:(end - 1), :, :] .|> DEFAULT_TYPE
        push!(data, u)
    end
    data = cat(data..., dims = 5);

    c_data = data[:, :, :, 1:(end - 1), :];
    y_data = data[:, :, :, 2:end, :];

    c_data = reshape(c_data, 128, 128, 2, num_trajectories * (num_steps - 1));
    y_data = reshape(y_data, 128, 128, 2, num_trajectories * (num_steps - 1));

    if normal_base_distribution
        return (;
            base = Distributions.Normal(0.0f0, 1.0f0),
            target = y_data,
            field_conditioning = c_data
        )
    end

    return (; base = c_data, target = y_data, field_conditioning = c_data)
end

function load_kolmogorov_data(config::Config.KolmogorovDataHyperparameters)
    return load_kolmogorov_data(
        config.num_trajectories,
        config.num_skip_steps,
        config.num_steps,
        config.start_step,
        config.normal_base_distribution
    )
end
