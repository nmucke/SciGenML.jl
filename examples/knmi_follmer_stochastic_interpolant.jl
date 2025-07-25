using SciGenML
import SciGenML.Models as Models
import SciGenML.Training as Training
import SciGenML.Sampling as Sampling
import SciGenML.Config as Config
import SciGenML.Utils as Utils
import SciGenML.TimeIntegrators as TimeIntegrators
import SciGenML.Plotting as Plotting
import SciGenML.Data as Data
import SciGenML.Preprocessing as Preprocessing
import Lux
using LuxCUDA
import Configurations
import Random
import Plots
import CUDA

CUDA.math_mode!(CUDA.FAST_MATH)

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
train_data, test_data = Data.load_data(config);

##### Preprocess data #####
data_preprocessor = Preprocessing.DataPreprocessor(
    Preprocessing.StandardScaler(train_data.base),
    Preprocessing.StandardScaler(train_data.target),
    Preprocessing.StandardScaler(train_data.field_conditioning)
);

train_data = data_preprocessor.transform(train_data);
test_data = (;
    base = data_preprocessor.base.transform(test_data.base),
    field_conditioning = data_preprocessor.field_conditioning.transform(test_data.field_conditioning)
);

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
SI_model =
    Training.train(SI_model, train_data, config; verbose = true, checkpoint = checkpoint);

##### Sample using model #####
num_gen_samples = 4;
num_steps = 50;
start_time = 1;
num_physical_steps = 5000;

test_samples = test_data.base[:, :, :, start_time:(start_time + num_physical_steps), 1];

init_condition = test_samples[:, :, :, 1:1] |> device;
init_condition = cat((init_condition for i in 1:num_gen_samples)..., dims = 4);

field_conditioning =
    test_data.field_conditioning[:, :, :, start_time:(start_time + num_physical_steps), 1];
field_conditioning = cat((field_conditioning for i in 1:num_gen_samples)..., dims = 5);

pred_trajectories = zeros(DEFAULT_TYPE, 64, 128, 1, num_physical_steps, num_gen_samples);
pred_trajectories[:, :, :, 1, :] =
    data_preprocessor.base.inverse_transform(init_condition |> cpu_dev);
iter = Utils.get_iter(num_physical_steps-1, true);
for i in iter
    init_condition, _st = Sampling.sample(
        SI_model,
        cat(init_condition |> device, field_conditioning[:, :, 2:2, i, :], dims = 3),
        num_steps;
        prior_samples = init_condition,
        verbose = false,
        stepper = TimeIntegrators.heun_step
    )

    sol = init_condition |> cpu_dev
    sol = data_preprocessor.base.inverse_transform(sol)
    pred_trajectories[:, :, :, i + 1, :] = sol
end

test_trajectories = data_preprocessor.base.inverse_transform(test_samples);
trajectory_list = [permutedims(test_trajectories[:, :, 1, 1:num_physical_steps], (2, 1, 3))];
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
true_temperature = Statistics.mean(test_trajectories, dims = (1, 2))[1, 1, 1, :];

Plots.plot(pred_temperature[:, 1], label = "SI Prediction")
Plots.plot!(true_temperature, label = "True")
Plots.savefig("knmi_temperature_$(num_steps).png")
