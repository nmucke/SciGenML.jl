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
NUM_SAMPLES = 5000
device = Lux.gpu_device();
cpu_dev = Lux.CPUDevice();
rng = Lux.Random.default_rng();

##### Load config #####
config =
    Configurations.from_toml(Config.Hyperparameters, "configs/2d_dense_flow_matching.toml");

unet = Architectures.UNet(1, 1, [16, 32, 64, 128], 1, 64, "constant");

##### Define generative model #####
FM_model = Models.FlowMatching(unet,);
FM_model = Utils.move_to_device(FM_model, device);

##### Get training data #####
y_data, _ = MLDatasets.MNIST(split = :train)[1:NUM_SAMPLES];
y_data = reshape(y_data, 28, 28, 1, :);
y_data = y_data .|> DEFAULT_TYPE;
y_data = NNlib.pad_zeros(y_data, (2, 2, 0, 0));

x_data_dist = Distributions.Normal(0.0f0, 1.0f0);
x_data = rand(rng, x_data_dist, (32, 32, 1, NUM_SAMPLES)) .|> DEFAULT_TYPE;

##### Train model #####
data = (base = x_data_dist, target = y_data);
FM_model = Training.train(FM_model, data, config; verbose = true);

##### Sample using model #####
fm_samples, st = Sampling.sample(
    FM_model,
    250;
    prior_samples = rand(rng, x_data_dist, (32, 32, 1, 8)) .|> DEFAULT_TYPE,
    num_samples = NUM_SAMPLES,
    verbose = true
);
fm_samples = fm_samples |> cpu_dev;

p = Plots.plot(Plots.heatmap(fm_samples[:, :, 1, 1]), Plots.heatmap(fm_samples[:, :, 1, 2]))
