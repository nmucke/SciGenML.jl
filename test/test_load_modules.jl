
@testitem "Submodule Loading" begin
    using SciGenML
    @test isdefined(SciGenML, :Models)
    @test isdefined(SciGenML, :Architectures)
    @test isdefined(SciGenML, :Layers)
    @test isdefined(SciGenML, :TimeIntegrators)
    @test isdefined(SciGenML, :Data)
    @test isdefined(SciGenML, :Sampling)
    @test isdefined(SciGenML, :Training)
    @test isdefined(SciGenML, :Config)
    @test isdefined(SciGenML, :Utils)
end
