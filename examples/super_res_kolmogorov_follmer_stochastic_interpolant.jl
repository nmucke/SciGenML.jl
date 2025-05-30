using SciGenML
import SciGenML.Models as Models
import SciGenML.Training as Training
import SciGenML.Sampling as Sampling
import SciGenML.Config as Config
import SciGenML.Utils as Utils
import SciGenML.TimeIntegrators as TimeIntegrators
import SciGenML.Plotting as Plotting
import SciGenML.Data as Data
import Lux
using LuxCUDA
import Configurations
import Random
import JLD2

device = Lux.gpu_device();
cpu_dev = Lux.CPUDevice();
rng = Lux.Random.default_rng();

# import ImageTransformations as IT
# import Plots 

# reduction_factor = 8

# function create_super_res_kolmogorov_data(reduction_factor::Int)
#     for i in 1:50
#         high_res_data = JLD2.load("data/kolmogorov_128/sim_$i.jld2")["u"]
#         high_res_data = high_res_data[2:end-1, 2:end-1, :, 1:3200]

#         low_res_data = zeros(128 ÷ reduction_factor, 128 ÷ reduction_factor, 2, 3200)
#         upscaled_data = zeros(128, 128, 2, 3200)
#         for time in 1:3200
#             for channel in 1:2
#                 img_small = IT.imresize(high_res_data[:, :, channel, time], ratio=1/reduction_factor)
#                 img_medium = IT.imresize(img_small, size(img_small).*reduction_factor)
#                 low_res_data[:, :, channel, time] = img_small
#                 upscaled_data[:, :, channel, time] = img_medium
#             end
#         end

#         JLD2.save(
#             "data/super_res_kolmogorov/sim_$i.jld2", 
#             Dict(
#                 "low_res" => low_res_data, 
#                 "upscaled" => upscaled_data,
#                 "high_res" => high_res_data
#             )
#         )
#     end
# end

# create_super_res_kolmogorov_data(reduction_factor)

##### Load config #####
config = Configurations.from_toml(
    Config.Hyperparameters,
    "configs/super_res_kolmogorov_stochastic_interpolant.toml"
);

##### Load data #####
data = Data.load_data(config; with_low_res = false);

##### Define generative model #####
SI_model = Models.get_model(config);
SI_model = Utils.move_to_device(SI_model, device);

##### Checkpointing #####
checkpoint_path = "checkpoints/super_res_kolmogorov_follmer_stochastic_interpolant";
checkpoint = Training.Checkpoint(checkpoint_path, config);

##### Train model #####
SI_model = Training.train(SI_model, data, config; verbose = true, checkpoint = checkpoint);

##### Sample using model #####

data = Data.load_data(config; with_low_res = true);

test_ids = [10, 100, 250, 400];
num_steps = 250;

prior_samples = data.base[:, :, :, test_ids];
pred_high_res, _st = Sampling.sample(
    SI_model,
    prior_samples,
    num_steps;
    prior_samples = prior_samples,
    verbose = true,
    stepper = TimeIntegrators.heun_step
);

pred_high_res = pred_high_res |> cpu_dev;
true_high_res = data.target[:, :, :, test_ids] |> cpu_dev;
low_res = data.low_res[:, :, :, test_ids] |> cpu_dev;

low_res = sqrt.(low_res[:, :, 1, :] .^ 2 + low_res[:, :, 2, :] .^ 2) |> cpu_dev;
linear_upscaling =
    sqrt.(prior_samples[:, :, 1, :] .^ 2 + prior_samples[:, :, 2, :] .^ 2) |> cpu_dev;
pred_high_res = sqrt.(pred_high_res[:, :, 1, :] .^ 2 + pred_high_res[:, :, 2, :] .^ 2);
true_high_res = sqrt.(true_high_res[:, :, 1, :] .^ 2 + true_high_res[:, :, 2, :] .^ 2);

import Plots

Plots.default(size = (1200, 1200))
Plots.plot(
    Plots.heatmap(low_res[:, :, 1], title = "Low Res 1", cbar = false),
    Plots.heatmap(low_res[:, :, 2], title = "Low Res 2", cbar = false),
    Plots.heatmap(low_res[:, :, 3], title = "Low Res 3", cbar = false),
    Plots.heatmap(low_res[:, :, 4], title = "Low Res 4", cbar = false),
    Plots.heatmap(linear_upscaling[:, :, 1], title = "Linear upscaling 1", cbar = false),
    Plots.heatmap(linear_upscaling[:, :, 2], title = "Linear upscaling 2", cbar = false),
    Plots.heatmap(linear_upscaling[:, :, 3], title = "Linear upscaling 3", cbar = false),
    Plots.heatmap(linear_upscaling[:, :, 4], title = "Linear upscaling 4", cbar = false),
    Plots.heatmap(pred_high_res[:, :, 1], title = "SI High Res 1", cbar = false),
    Plots.heatmap(pred_high_res[:, :, 2], title = "SI High Res 2", cbar = false),
    Plots.heatmap(pred_high_res[:, :, 3], title = "SI High Res 3", cbar = false),
    Plots.heatmap(pred_high_res[:, :, 4], title = "SI High Res 4", cbar = false),
    Plots.heatmap(true_high_res[:, :, 1], title = "True High Res 1", cbar = false),
    Plots.heatmap(true_high_res[:, :, 2], title = "True High Res 2", cbar = false),
    Plots.heatmap(true_high_res[:, :, 3], title = "True High Res 3", cbar = false),
    Plots.heatmap(true_high_res[:, :, 4], title = "True High Res 4", cbar = false),
    layout = (4, 4)
)
