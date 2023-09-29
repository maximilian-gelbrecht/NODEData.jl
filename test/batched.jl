@testset "MultiTrajectoryBatchedNODEDataloader" begin 
    # constrcutor 
    trajectories = [(1:40, rand(5,5,40)) for i=1:4]

    train = NODEData.MultiTrajectoryBatchedNODEDataloader(trajectories, 3)

    @test train[1][1] == 1:3 
    @test train[2][1] == 2:4
    @test train[end][1] == 38:40

    @test train[1][2][:,:,1,:] == trajectories[1][2][:,:,1:3]
    @test train[1][2][:,:,2,:] == trajectories[2][2][:,:,1:3]

    train, valid = NODEData.MultiTrajectoryBatchedNODEDataloader(trajectories, 3, valid_set=0.5)

    @test typeof(valid[1][2]) <: AbstractArray 

    # single_trajectory_from_batched
    traj = NODEData.single_trajectory_from_batched(train,10)
    @test traj[1] == 1:10 
    @test traj[2][:,:,1,:] == trajectories[1][2][:,:,1:10]
    @test traj[2][:,:,2,:] == trajectories[1][2][:,:,1:10]
end 

@testset "SingleTrajectoryBatchedOSADataloader" begin 

    t = 1:51
    x = rand(5,5,51)

    # constructor 
    train = NODEData.SingleTrajectoryBatchedOSADataloader(x, t, 5)

    # indexing 
    @test train[1][1][1,:] == 1:2 
    @test train[1][1][2,:] == 2:3 
    @test train[2][1][1,:] == 6:7
    @test train[end][1][end,:] == 50:51 

    @test train[1][2][:,:,1,1:2] == x[:,:,1:2]
    @test train[1][2][:,:,2,1:2] == x[:,:,2:3]
    @test train[2][2][:,:,1,1:2] == x[:,:,6:7]
    @test train[end][2][:,:,end,:] == x[:,:,50:51]

    # other base functions ;
    @test length(train) == 10 

    train, valid = NODEData.SingleTrajectoryBatchedOSADataloader(x, t, 5, valid_set =0.5)

    @test typeof(valid[1][2]) <: AbstractArray 

    # get_trajectory 
    traj = NODEData.get_trajectory(train, 10)
    @test traj[1] == 1:10 
    @test traj[2] == x[:,:,1:10] 
end