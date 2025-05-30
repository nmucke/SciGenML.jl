using SciGenML
import SciGenML.Models as Models
import SciGenML.Training as Training
import SciGenML.Sampling as Sampling
import SciGenML.Config as Config
import SciGenML.Utils as Utils
import SciGenML.TimeIntegrators as TimeIntegrators
import SciGenML.Plotting as Plotting
import SciGenML.Data as Data
using LuxCUDA
import Configurations
import Random

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

##### Define generative model #####
SI_model = Models.get_model(config);
SI_model = Utils.move_to_device(SI_model, device);

##### Train model #####
SI_model = Training.train(SI_model, data, config; verbose = true);

##### Sample using model #####
num_gen_samples = 4;
num_steps = 100;
num_physical_steps = 50;

init_condition = data.base[:, :, :, 1:1];
init_condition = init_condition |> device;
init_condition = cat((init_condition for i in 1:num_gen_samples)..., dims = 4);

pred_trajectories = []
iter = Utils.get_iter(num_physical_steps, true)
for i in iter
    init_condition, _st = Sampling.sample(
        SI_model,
        (init_condition,),
        num_steps;
        prior_samples = init_condition,
        verbose = false
    )
    SI_model.st = _st
    push!(pred_trajectories, init_condition |> cpu_dev)
end
pred_trajectories = stack(pred_trajectories, dims = 4);

Plotting.animate_velocity_magitude(
    [pred_trajectories[:, :, :, :, i] for i in 1:num_gen_samples],
    "kolmogorov_animation",
    ["Simulation $i" for i in 1:num_gen_samples];
    velocity_channels = (1, 2)
)
