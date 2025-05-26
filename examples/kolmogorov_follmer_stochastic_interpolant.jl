using SciGenML
import SciGenML.Architectures as Architectures
import SciGenML.Models as Models
import SciGenML.Training as Training
import SciGenML.Sampling as Sampling
import SciGenML.Config as Config
import SciGenML.Utils as Utils
import SciGenML.TimeIntegrators as TimeIntegrators
import SciGenML.Layers as Layers
import Lux
using LuxCUDA
import Configurations
import Distributions
import Plots
import Random
import MLDatasets
import NNlib
NUM_SAMPLES = 1500

device = Lux.gpu_device();
cpu_dev = Lux.CPUDevice();
rng = Lux.Random.default_rng();

##### Load config #####
config = Configurations.from_toml(
    Config.Hyperparameters,
    "configs/kolmogorov_stochastic_interpolant.toml"
);

unet = Architectures.UNet(
    config.architecture.in_channels,
    config.architecture.out_channels,
    config.architecture.hidden_channels,
    config.architecture.time_embedding_dim,
    config.architecture.padding,
    config.architecture.field_in_conditioning_dim,
    config.architecture.field_hidden_conditioning_dim,
    config.architecture.field_conditioning_combination
);

##### Define generative model #####
SI_model = Models.FollmerStochasticInterpolant(unet);
SI_model = Utils.move_to_device(SI_model, device);

##### Get training data #####
y_data = 0
c_data = 0
x_data = c_data;

##### Train model #####
data = (base = x_data, target = y_data, field_conditioning = c_data);
SI_model = Training.train(SI_model, data, config; verbose = true);

##### Sample using model #####
num_gen_samples = 8
num_steps = 50
gen_c_data = x_data[:, :, :, 1:num_gen_samples];
si_samples, st = Sampling.sample(
    Models.Stochastic(),
    SI_model,
    (gen_c_data,),
    num_steps;
    prior_samples = x_data[:, :, :, 1:num_gen_samples],
    num_samples = NUM_SAMPLES,
    verbose = true,
    stepper = TimeIntegrators.euler_maruyama_step,
    diffusion_fn = t -> 5.5f0 .* (1.0f0 .- t)
);
si_samples = si_samples |> cpu_dev;

heatmaps = [
    Plots.heatmap(si_samples[:, :, 1, i], title = "$(gen_c_data[1, 1, 1, i])") for i in 1:8
];
p = Plots.plot(heatmaps...)
