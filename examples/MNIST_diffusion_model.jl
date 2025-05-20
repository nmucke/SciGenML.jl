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

NUM_SAMPLES = 500000
device = Lux.gpu_device();
cpu_dev = Lux.CPUDevice();
rng = Lux.Random.default_rng();

##### Load config #####
config =
    Configurations.from_toml(Config.Hyperparameters, "configs/unet_diffusion_model.toml");

unet = Architectures.UNet(
    config.architecture.in_channels,
    config.architecture.out_channels,
    config.architecture.hidden_channels,
    config.architecture.in_conditioning_dim,
    config.architecture.time_embedding_dim,
    config.architecture.padding
);

##### Define generative model #####
diffusion_model = Models.ScoreBasedDiffusionModel(unet);
diffusion_model = Utils.move_to_device(diffusion_model, device);

##### Get training data #####
y_data, targets = MLDatasets.MNIST(split = :train)[:];
y_data = y_data[
    :,
    :,
    (targets .== 1) .| (targets .== 2) .| (targets .== 3) .| (targets .== 4) .| (targets .== 5)
];
NUM_SAMPLES = size(y_data, 3);
y_data = reshape(y_data, 28, 28, 1, :);
y_data = y_data .|> DEFAULT_TYPE;
y_data = NNlib.pad_zeros(y_data, (2, 2, 0, 0));
y_data = permutedims(y_data, (2, 1, 3, 4));

x_data_dist = Distributions.Normal(0.0f0, 1.0f0);
x_data = rand(rng, x_data_dist, (32, 32, 1, NUM_SAMPLES)) .|> DEFAULT_TYPE;

##### Train model #####
data = (base = x_data_dist, target = y_data);
diffusion_model = Training.train(diffusion_model, data, config; verbose = true);

##### Sample using model #####
diffusion_samples, st = Sampling.sample(
    diffusion_model,
    50;
    prior_samples = rand(rng, x_data_dist, (32, 32, 1, 8)) .|> DEFAULT_TYPE,
    num_samples = NUM_SAMPLES,
    verbose = true
);
diffusion_model.st = st;
diffusion_samples = diffusion_samples |> cpu_dev;
diffusion_samples = clamp.(diffusion_samples, 0.0f0, 1.0f0);

p = Plots.plot(
    Plots.heatmap(diffusion_samples[:, :, 1, 1]),
    Plots.heatmap(diffusion_samples[:, :, 1, 2])
)
