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
import Plots

LOAD_CHECKPOINT = true;
CHECKPOINT_PATH = "checkpoints/knmi_stochastic_interpolant";

device = Lux.gpu_device();
cpu_dev = Lux.CPUDevice();
rng = Lux.Random.default_rng();

##### Load config #####
config = Configurations.from_toml(
    Config.Hyperparameters,
    "configs/knmi_stochastic_interpolant.toml"
);

##### Load data #####
data = Data.load_data(config);

##### Checkpoint #####
checkpoint = Training.Checkpoint(CHECKPOINT_PATH, config; create_new = !LOAD_CHECKPOINT);
if LOAD_CHECKPOINT
    train_state = Training.load_train_state(checkpoint);

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
num_steps = 20;
start_time = 50000;
num_physical_steps = 10000;

test_data = data.target[:, :, :, start_time:(start_time + num_physical_steps)];

init_condition = test_data[:, :, :, 1:1] |> device;
init_condition = cat((init_condition for i in 1:num_gen_samples)..., dims = 4);

field_conditioning =
    data.field_conditioning[:, :, :, start_time:(start_time + num_physical_steps)];
field_conditioning = cat((field_conditioning for i in 1:num_gen_samples)..., dims = 5);

pred_trajectories = zeros(Float32, 64, 128, 1, num_physical_steps, num_gen_samples);
pred_trajectories[:, :, :, 1, :] = init_condition |> cpu_dev;
iter = Utils.get_iter(num_physical_steps-1, true);
for i in iter
    init_condition, _st = Sampling.sample(
        SI_model,
        cat(init_condition, field_conditioning[:, :, 2:2, i, :], dims = 3),
        num_steps;
        prior_samples = init_condition,
        verbose = false,
        stepper = TimeIntegrators.heun_step
    )
    # SI_model.st = _st
    pred_trajectories[:, :, :, i + 1, :] = init_condition |> cpu_dev
end

trajectory_list = [permutedims(test_data[:, :, 1, 1:num_physical_steps], (2, 1, 3))];
for i in 1:num_gen_samples
    push!(trajectory_list, permutedims(pred_trajectories[:, :, 1, :, i], (2, 1, 3)));
end

# trajectory_list = [permutedims(test_data[:, :, 1, 1:num_physical_steps], (2, 1, 3)),]
Plotting.animate_field(
    trajectory_list,
    "knmi_animation_$(num_steps)",
    ["$(i == 1 ? "True" : "Prediction $i")" for i in 1:(num_gen_samples + 1)];
    framerate = 25
)

import Statistics
pred_temperature = Statistics.mean(pred_trajectories, dims = (1, 2))[1, 1, 1, :, :];
true_temperature = Statistics.mean(test_data, dims = (1, 2))[1, 1, 1, :];

Plots.plot(pred_temperature[:, 1], label = "SI Prediction")
Plots.plot!(true_temperature, label = "True")
Plots.savefig("knmi_temperature_$(num_steps).png")
