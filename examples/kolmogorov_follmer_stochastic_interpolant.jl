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

LOAD_CHECKPOINT = true;
CHECKPOINT_PATH = "checkpoints/linear_kolmogorov_stochastic_interpolant.jld2";

device = Lux.gpu_device();
cpu_dev = Lux.CPUDevice();
rng = Lux.Random.default_rng();

##### Load config #####
config = Configurations.from_toml(
    Config.Hyperparameters,
    "configs/kolmogorov_stochastic_interpolant.toml"
);

##### Load data #####
data = Data.load_data(config);

##### Checkpoint #####
checkpoint = Training.Checkpoint(CHECKPOINT_PATH, config);
if LOAD_CHECKPOINT
    train_state = Training.load_train_state(checkpoint) |> cpu_dev;

    config = checkpoint.config;
end

##### Define generative model #####
SI_model = Models.get_model(config);

if LOAD_CHECKPOINT
    SI_model.ps = (; velocity = train_state["ps"]);
    SI_model.st = (; velocity = train_state["st"]);
end;

SI_model = Utils.move_to_device(SI_model, device);

##### Train model #####
SI_model = Training.train(SI_model, data, config; verbose = true, checkpoint = checkpoint);

##### Sample using model #####
num_gen_samples = 4;
num_steps = 50;
num_physical_steps = 25;

test_data = data.target[:, :, :, 1:num_physical_steps];

init_condition = test_data[:, :, :, 1:1] |> device;
init_condition = cat((init_condition for i in 1:num_gen_samples)..., dims = 4);

pred_trajectories = zeros(DEFAULT_TYPE, 128, 128, 2, num_physical_steps, num_gen_samples);
pred_trajectories[:, :, :, 1, :] = init_condition |> cpu_dev;
iter = Utils.get_iter(num_physical_steps-1, true);
for i in iter
    init_condition, _st = Sampling.sample(
        SI_model,
        init_condition,
        num_steps;
        prior_samples = init_condition,
        verbose = false
    )
    # SI_model.st = _st
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
