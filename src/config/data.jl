
"""
    KolmogorovDataHyperparameters

    Hyperparameters for the kolmogorov data.
"""
Configurations.@option "kolmogorov" struct KolmogorovDataHyperparameters
    train_trajectories::Vector{Int}
    num_skip_steps::Int
    train_num_steps::Int
    start_step::Int
    normal_base_distribution::Bool
end

"""
    SuperResKolmogorovDataHyperparameters

    Hyperparameters for the super res kolmogorov data.
"""
Configurations.@option "super_res_kolmogorov" struct SuperResKolmogorovDataHyperparameters
    trajectories::Vector{Int}
    num_skip_steps::Int
    train_num_steps::Int
    start_step::Int
    normal_base_distribution::Bool
end

"""
    KNMIDataHyperparameters

    Hyperparameters for the KNMI data.
"""
Configurations.@option "knmi" struct KNMIDataHyperparameters
    train_num_steps::Int
    train_trajectories::Vector{Int}
    num_skip_steps::Int
    test_num_steps::Int
    test_trajectories::Vector{Int}
end
