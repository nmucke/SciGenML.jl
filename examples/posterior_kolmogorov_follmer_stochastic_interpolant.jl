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
import Distributions
import LinearAlgebra
using SparseArrays
using CUDA

device = Lux.gpu_device();
cpu_dev = Lux.CPUDevice();
rng = Lux.Random.default_rng();

SIGMA = 0.01f0

function observation_operator(x)
    idx1 = 1:5:128
    idx2 = 1:5:128
    idx3 = 1:1
    inds = vec([CartesianIndex(i, j, k) for i in idx1, j in idx2, k in idx3])
    inds = LinearIndices(size(x)[1:3])[inds]
    x_flat = reshape(x, :, size(x, 4))
    out = x_flat[inds, :]
    out = reshape(out, length(idx1), length(idx2), length(idx3), size(x, 4))
    return out
end

function log_likelihood_fn(x, y; mu = 0.0f0, sigma = SIGMA)
    out = sum((y .- x) .^ 2)

    return -0.5f0 ./ (sigma .^ 2) .* out
end

x = Random.randn(DEFAULT_TYPE, 128, 128, 2, 1) |> device;
y = Random.randn(DEFAULT_TYPE, 26, 26, 1, 1) |> device;
observation_operator(x)
log_likelihood_fn(x, y)

##### Load config #####
config = Configurations.from_toml(
    Config.Hyperparameters,
    "configs/kolmogorov_stochastic_interpolant.toml"
);

##### Load data #####
data = Data.load_data(config);

##### Define generative model #####
SI_model = Models.get_model(config);

##### Checkpoint #####
checkpoint_path = "checkpoints/kolmogorov_stochastic_interpolant.jld2";
checkpoint = Training.Checkpoint(checkpoint_path, config);
train_state = Training.load_train_state(checkpoint);

SI_model.ps = (; velocity = train_state["ps"]);
SI_model.st = (; velocity = train_state["st"]);

SI_model = Utils.move_to_device(SI_model, device);

##### Sample using model #####
num_gen_samples = 4;
num_steps = 100;
num_physical_steps = 3;

test_data = data.target[:, :, :, 1:num_physical_steps];

init_condition = test_data[:, :, :, 1:1] |> device;
init_condition = cat((init_condition for i in 1:num_gen_samples)..., dims = 4);

pred_trajectories = zeros(DEFAULT_TYPE, 128, 128, 2, num_physical_steps, num_gen_samples);
pred_trajectories[:, :, :, 1, :] = init_condition |> cpu_dev;
iter = Utils.get_iter(num_physical_steps-1, true);
for i in iter
    noise = Random.randn(DEFAULT_TYPE, 26, 26, 1, 1) .* SIGMA |> device;
    obs = test_data[:, :, :, i:i] |> device
    obs = observation_operator(obs) .+ noise;

    obs = cat(obs, obs, obs, obs, dims = 4);

    init_condition, _st = Sampling.posterior_sample(
        SI_model,
        init_condition,
        obs,
        observation_operator,
        log_likelihood_fn,
        num_steps;
        prior_samples = init_condition,
        verbose = false,
        diffusion_fn = t -> 0.1f0 .- t
    )
    pred_trajectories[:, :, :, i + 1, :] = init_condition |> cpu_dev
end

trajectory_list = [test_data[:, :, :, 1:num_physical_steps]];
for i in 1:num_gen_samples
    push!(trajectory_list, pred_trajectories[:, :, :, :, i]);
end

Plotting.animate_velocity_magitude(
    trajectory_list,
    "kolmogorov_animation",
    ["$(i == 1 ? "True" : "Prediction $i")" for i in 1:(num_gen_samples + 1)];
    velocity_channels = (1, 2)
)
