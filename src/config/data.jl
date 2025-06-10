
"""
    KolmogorovDataHyperparameters

    Hyperparameters for the kolmogorov data.
"""
Configurations.@option "kolmogorov" struct KolmogorovDataHyperparameters
    num_trajectories::Int
    num_skip_steps::Int
    num_steps::Int
    start_step::Int
    normal_base_distribution::Bool
end

"""
    SuperResKolmogorovDataHyperparameters

    Hyperparameters for the super res kolmogorov data.
"""
Configurations.@option "super_res_kolmogorov" struct SuperResKolmogorovDataHyperparameters
    num_trajectories::Int
    num_skip_steps::Int
    num_steps::Int
    start_step::Int
    normal_base_distribution::Bool
end

"""
    KNMIDataHyperparameters

    Hyperparameters for the KNMI data.
"""
Configurations.@option "knmi" struct KNMIDataHyperparameters
    num_steps::Int
    num_trajectories::Int
    num_skip_steps::Int
end
