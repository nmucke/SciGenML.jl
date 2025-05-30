"""
    load_super_res_kolmogorov_data(;
        num_trajectories = 5,
        num_skip_steps = 5,
        num_steps = 100,
        start_step = 500
    )
    Load the super-resolution Kolmogorov data.

    Outputs a named tuple with the following fields:
    - `base`: The base data. [H, W, C, (T-1)*N]
    - `target`: The target data. [H, W, C, (T-1)*N]
    - `field_conditioning`: The field conditioning data. [H, W, C, (T-1)*N]
"""
function load_super_res_kolmogorov_data(
    num_trajectories = 5,
    num_skip_steps = 5,
    num_steps = 100,
    start_step = 500,
    normal_base_distribution = false;
    with_low_res = false
)
    high_res_data = []
    upscaled_data = []
    low_res_data = []
    for i in 1:num_trajectories
        data = JLD2.load("data/super_res_kolmogorov/sim_$(i).jld2")
        high_res = data["high_res"][
            :,
            :,
            :,
            start_step:num_skip_steps:(start_step + num_skip_steps * num_steps - 1)
        ]
        high_res = high_res .|> DEFAULT_TYPE
        push!(high_res_data, high_res)

        upscaled = data["upscaled"][
            :,
            :,
            :,
            start_step:num_skip_steps:(start_step + num_skip_steps * num_steps - 1)
        ]
        upscaled = upscaled .|> DEFAULT_TYPE
        push!(upscaled_data, upscaled)

        low_res = data["low_res"][
            :,
            :,
            :,
            start_step:num_skip_steps:(start_step + num_skip_steps * num_steps - 1)
        ]
        low_res = low_res .|> DEFAULT_TYPE
        push!(low_res_data, low_res)
    end
    high_res_data = cat(high_res_data..., dims = 5);
    upscaled_data = cat(upscaled_data..., dims = 5);
    low_res_data = cat(low_res_data..., dims = 5);

    high_res_data = reshape(high_res_data, 128, 128, 2, num_trajectories * num_steps);
    upscaled_data = reshape(upscaled_data, 128, 128, 2, num_trajectories * num_steps);
    low_res_data = reshape(
        low_res_data,
        size(low_res_data, 1),
        size(low_res_data, 2),
        2,
        num_trajectories * num_steps
    );

    if normal_base_distribution
        base = Distributions.Normal(0.0f0, 1.0f0)
    else
        base = upscaled_data
    end

    if with_low_res
        return (;
            base = base,
            target = high_res_data,
            field_conditioning = upscaled_data,
            low_res = low_res_data
        )
    else
        return (; base = base, target = high_res_data, field_conditioning = upscaled_data)
    end
end

function load_super_res_kolmogorov_data(
    config::Config.SuperResKolmogorovDataHyperparameters;
    kwargs...
)
    return load_super_res_kolmogorov_data(
        config.num_trajectories,
        config.num_skip_steps,
        config.num_steps,
        config.start_step,
        config.normal_base_distribution;
        kwargs...
    )
end
