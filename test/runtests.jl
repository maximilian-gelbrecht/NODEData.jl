using NODEData
using Test
using OrdinaryDiffEq

@testset "NODEDataloader" begin

    f(u,p,t) = 1.01*u
    u0 = 1/2
    tspan = (0.0,10.0)
    prob = ODEProblem(f,u0,tspan)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

    # interpolate
    data = NODEDataloader(sol, 20, dt=0.2)

    @test size(data[1][2])[end] == 20
    @test size(data[1][2])[end] == 20
    @test data[1][1] == 0:0.2:3.8
    @test data[1][2][2:end] == data[2][2][1:end-1]
    @test data[end-1][2][2:end] == data[end][2][1:end-1]


    # valid set

    train, valid = NODEDataloader(sol, 20, dt=0.2, valid_set=0.2)


    # original data
    data = NODEDataloader(sol, 10)

    @test size(data[1][2])[end] == 10
    @test data[1][1] == sol.t[1:10]
    @test data[1][2] == Array(sol)[1:10]
    @test data[1][2][2:end] == data[2][2][1:end-1]
    @test data[end-1][2][2:end] == data[end][2][1:end-1]



end

# TBD: better tests
@testset "LargeNODEDataloader" begin

    f(u,p,t) = 1.01*u
    u0 = 1/2
    tspan = (0.0,50.0)
    prob = ODEProblem(f,u0,tspan)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

    # interpolate
    data = LargeNODEDataloader(sol, 10, 5, "test", dt=0.2)

    @test (data[1])[1][1] == 0.0:0.2:0.8


    delete(data)

    data, valid = LargeNODEDataloader(sol, 10, 5, "test", dt=0.2, valid_set=true)

    @test typeof(valid[1][2]) <: AbstractArray

    delete(data)
end
