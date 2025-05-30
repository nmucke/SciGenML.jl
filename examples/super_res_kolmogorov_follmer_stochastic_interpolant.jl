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

device = Lux.gpu_device();
cpu_dev = Lux.CPUDevice();
rng = Lux.Random.default_rng();

##### Load config #####
config = Configurations.from_toml(
    Config.Hyperparameters,
    "configs/super_res_kolmogorov_stochastic_interpolant.toml"
);

##### Load data #####
data = Data.load_data(config);

##### Define generative model #####
SI_model = Models.get_model(config);
SI_model = Utils.move_to_device(SI_model, device);

##### Train model #####
SI_model = Training.train(SI_model, data, config; verbose = true);

##### Sample using model #####
test_ids = [1000, 3000, 5000, 7000];
num_steps = 100;
num_physical_steps = 50;

prior_samples = data.base[:, :, :, test_ids];
pred_high_res, _st = Sampling.sample(
    SI_model,
    prior_samples,
    num_steps;
    prior_samples = prior_samples,
    verbose = true
);

pred_high_res = pred_high_res |> cpu_dev;
true_high_res = data.target[:, :, :, test_ids] |> cpu_dev;

low_res = sqrt.(prior_samples[:, :, 1, :] .^ 2 + prior_samples[:, :, 2, :] .^ 2) |> cpu_dev;
pred_high_res = sqrt.(pred_high_res[:, :, 1, :] .^ 2 + pred_high_res[:, :, 2, :] .^ 2);
true_high_res = sqrt.(data.target[:, :, 1, :] .^ 2 + data.target[:, :, 2, :] .^ 2);

import Plots

Plots.default(size = (1200, 600))
Plots.plot(
    Plots.heatmap(low_res[:, :, 1], title = "Low Res 1"),
    Plots.heatmap(low_res[:, :, 2], title = "Low Res 2"),
    Plots.heatmap(low_res[:, :, 3], title = "Low Res 3"),
    Plots.heatmap(low_res[:, :, 4], title = "Low Res 4"),
    Plots.heatmap(pred_high_res[:, :, 1], title = "Predicted High Res 1"),
    Plots.heatmap(pred_high_res[:, :, 2], title = "Predicted High Res 2"),
    Plots.heatmap(pred_high_res[:, :, 3], title = "Predicted High Res 3"),
    Plots.heatmap(pred_high_res[:, :, 4], title = "Predicted High Res 4"),
    Plots.heatmap(true_high_res[:, :, 1], title = "True High Res 1"),
    Plots.heatmap(true_high_res[:, :, 2], title = "True High Res 2"),
    Plots.heatmap(true_high_res[:, :, 3], title = "True High Res 3"),
    Plots.heatmap(true_high_res[:, :, 4], title = "True High Res 4"),
    layout = (3, 4)
)
