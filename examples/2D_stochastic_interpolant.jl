using SciGenML
import SciGenML.Architectures as Architectures
import SciGenML.Models as Models
import SciGenML.Training as Training
import SciGenML.Sampling as Sampling
import SciGenML.Config as Config
import SciGenML.Utils as Utils
import SciGenML.TimeIntegrators as TimeIntegrators
import Lux
import Configurations
import Distributions
import Plots
import Random

NUM_SAMPLES = 25000
device = Lux.gpu_device();
cpu_dev = Lux.CPUDevice();
rng = Lux.Random.default_rng();

##### Load config #####
config = Configurations.from_toml(Config.Hyperparameters, "configs/2d_dense_SI.toml");

##### Define generative model #####
SI_model = Models.StochasticInterpolant(config, Models.Stochastic());
SI_model = Utils.move_to_device(SI_model, cpu_dev);

##### Define training data #####
x_data_dist = Distributions.Normal(0.0, 1.0);
x_data = rand(rng, x_data_dist, (2, NUM_SAMPLES)) .|> DEFAULT_TYPE;

y_data_dist = Distributions.MixtureModel(Distributions.Normal[
    Distributions.Normal(-3.0, 1.0),
    Distributions.Normal(3.0, 1.0)
]);
y_data = rand(rng, y_data_dist, (2, NUM_SAMPLES)) .|> DEFAULT_TYPE;

p = Plots.plot(
    Plots.histogram2d(
        x_data[1, :],
        x_data[2, :],
        bins = 100,
        normalize = :density,
        label = "x_data"
    ),
    Plots.histogram2d(
        y_data[1, :],
        y_data[2, :],
        bins = 100,
        normalize = :density,
        label = "y_data"
    )
)

##### Train model #####
data = (base = x_data_dist, target = y_data);
SI_model = Training.train(SI_model, data, config; verbose = true);

##### Sample using model #####
si_samples, st = Sampling.sample(
    Models.Stochastic(),
    SI_model,
    100;
    prior_samples = rand(rng, x_data_dist, (2, NUM_SAMPLES)) .|> DEFAULT_TYPE,
    num_samples = NUM_SAMPLES,
    verbose = true
);
si_samples = si_samples |> cpu_dev

p = Plots.plot(
    Plots.histogram2d(
        si_samples[1, :],
        si_samples[2, :],
        bins = 100,
        normalize = :density,
        label = "SI samples"
    ),
    Plots.histogram2d(
        y_data[1, :],
        y_data[2, :],
        bins = 100,
        normalize = :density,
        label = "Target samples"
    )
)
