
@testitem "Submodule Loading" begin
    using SciGenML
    @test isdefined(SciGenML, :Models)
    @test isdefined(SciGenML, :NeuralNetworkArchitectures)
    @test isdefined(SciGenML, :NeuralNetworkLayers)
    @test isdefined(SciGenML, :TimeIntegrators)
    @test isdefined(SciGenML, :Data)
    @test isdefined(SciGenML, :Sampling)
    @test isdefined(SciGenML, :Training)
end
