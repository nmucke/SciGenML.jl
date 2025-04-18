import SciGenML.TimeIntegrators as TimeIntegrators


model = TimeIntegrators.lol()


using Test
@test import SciGenML.TimeIntegrators as TimeIntegrators

@test begin
    import SciGenML.NeuralNetworkArchitectures
    @test isdefined(SciGenML, :NeuralNetworkArchitectures)
end

using Test
using SciGenML
# import SciGenML.NeuralNetworkArchitectures
@test isdefined(SciGenMLs, :NeuralNetworkArchitecturess)

@test isdefined(Main, :SciGenML)

