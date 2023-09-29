using EllipsisNotation

abstract type AbstractBatchedNODEDataloader{T,U,N} <: AbstractNODEDataloader{T,U,N} end 

"""
    MultiTrajectoryBatchedNODEDataloader(trajectories::AbstractVector, N_length::Integer; valid_set=nothing, GPU::Union{Bool, Nothing}=nothing) where {T,U,N} 
    
* `trajectories` is a `Vector` of trajectories/tuples `(t, x(t))`, for the batched NODEDataloader it is assumed that all `t` are equal, and only the first one is used. The constructor enforces the equal length, but only warns about non-equal elements

Actually the routine just wraps around [`NODEDataloader`](@ref) but concatanates the `trajectories` in such a way that it works with it. 
"""
function MultiTrajectoryBatchedNODEDataloader(trajectories::AbstractVector, N_length::Integer; valid_set=nothing, GPU::Union{Bool, Nothing}=nothing)
    
    t = trajectories[1][1]

    for i in eachindex(trajectories)
        @assert length(trajectories[i][1]) == length(t)
        if t != trajectories[i][1]
            @warn "Unequal time axes in NODEDataloader, watch out!"
        end
    end

    data = cat([insert_dim(trajectories[i][2],ndims(trajectories[i][2])) for i=1:length(trajectories)]..., dims=ndims(trajectories[1][2]))

    NODEDataloader(data, t, N_length; valid_set=valid_set, GPU=GPU)
end 

function insert_dim(A, i_dim)
    s = [size(A)...]
    insert!(s, i_dim, 1)
    return reshape(A, s...)
end

"""
    single_trajectory_from_batched(data::NODEDataloader, N::Int)

Returns a single trajectory (t,x(t)) fomr the dataloader but repeats it along the N_batch dimension. Useful for testing the performance of methods that can't change the batch size dynamically. 
"""
function single_trajectory_from_batched(data::NODEDataloader, N::Int)
    s = [1 for i=1:(ndims(data.data)-2)]
    return (data.t[1:N], repeat(data.data[..,1:1,1:N],s...,size(data.data, ndims(data.data) -1), 1))
end

struct SingleTrajectoryBatchedOSADataloader{T<:AbstractArray,U<:AbstractVector,N<:Integer} <: AbstractBatchedNODEDataloader{T,U,N}
    data::T
    t::U
    N::N
    N_batch::N
end 

"""
    SingleTrajectoryBatchedOSADataloader(data::AbstractArray{T,N}, t::AbstractArray{U,1}, N_batch::Int=1; valid_set=nothing, GPU::Union{Bool, Nothing}=nothing) where {T,U,N} 

Prepares a single trajectory batches one-step-ahead dataloader. When indexed it returns tuples `(t, x)` which are batches of 2-element trajectories, so that they contain the initial condition and one step ahead of the dynamical system. The batches are in format (N_dims ... x N_batch x N_t). `t` contains the time information of each of these trajectory snippets. 

# Inputs 

* `data`: Trajectory (N_dims .... x N_t)
* `t`: time steps of the trajectory 
"""
function SingleTrajectoryBatchedOSADataloader(data::AbstractArray{T,N}, t::AbstractArray{U,1}, N_batch::Int=1; valid_set=nothing, GPU::Union{Bool, Nothing}=nothing) where {T,U,N} 

    set_gpu(typeof(data) <: CuArray ? true : false, GPU)

    if isnothing(valid_set)
        return SingleTrajectoryBatchedOSADataloader(_prepare_singletrajectory_batched(data, t, N_batch)..., N_batch)
    else 
        @assert 0 <= valid_set < 1 "Valid_set should be ∈ [0,1]"

        N_t = length(t)
        N_t_valid = Int(floor(valid_set*N_t))
        N_t_train = N_t - N_t_valid

        return SingleTrajectoryBatchedOSADataloader(_prepare_singletrajectory_batched(data[..,1:N_t_train], t[1:N_t_train], N_batch)..., N_batch), SingleTrajectoryBatchedOSADataloader(_prepare_singletrajectory_batched(data[..,1:N_t_train], t[1:N_t_train], N_batch)..., N_batch)
    end
end

function _prepare_singletrajectory_batched(data::AbstractArray{T,N}, t::AbstractArray{U,1}, N_batch::Int=1) where {T,U,N}

    N_N = div(length(t)-1, N_batch)
    if (length(t)-1) % N_batch != 0
        @warn "Not all data points are used, N_batch is not a divisor of the length of the trajectory"
    end

    dev_array_type = DeviceArrayType()
    batched_x = dev_array_type{dev_array_type{T,N+1}, 1}(undef, N_N) 
    batched_t = dev_array_type{Array{T,2}}(undef, N_N)
    
    i = 0
    for i_t=1:N_batch:(length(t)-N_batch)
        i += 1 
        # x
        batched_x[i] = DeviceArray(cat([insert_dim(data[..,i_t+i_b:i_t+i_b+1], ndims(data)) for i_b=0:(N_batch-1)]..., dims=ndims(data)))
        
        # t
        batched_t[i] = Array(cat([insert_dim(t[i_t+i_b:i_t+i_b+1],1) for i_b=0:(N_batch-1)]..., dims=1))      
    end
        
    return (batched_x, batched_t, N_N)
end 

function Base.getindex(iter::SingleTrajectoryBatchedOSADataloader{T,U,N}, i::Integer) where {T,U,N}
    @assert 0 < i <= iter.N
    return (iter.t[i], iter.data[i])
end 

Base.eltype(iter::SingleTrajectoryBatchedOSADataloader) = eltype(iter.data[1])

"""
    get_trajectory(data::SingleTrajectoryBatchedOSADataloader{T,U,N}, N)

Returns a (t, x(t)) tuple of length `N`.
"""
function get_trajectory(data::NODEData.SingleTrajectoryBatchedOSADataloader, N) 
    @assert N < data.N_batch * data.N

    N_batch = data.N_batch

    x_trajectory = DeviceArrayType(){eltype(data), ndims(data.data[1])-1}(undef, size(data.data[1])[1:end-2]..., N)
    t_trajectory = Array{eltype(data), 1}(undef, N)

    # initial state 
    t_trajectory[1] = data[1][1][1,1]
    x_trajectory[..,1] = data[1][2][..,1,1]

    for i=1:(N-1) 
        i_dataloader = div(i-1, N_batch) + 1
        i_batch = i % N_batch 
        i_batch = i_batch == 0 ? N_batch : i_batch

        t_trajectory[i+1] = data[i_dataloader][1][i_batch,2] 
        x_trajectory[..,i+1] = data[i_dataloader][2][..,i_batch,2]
    end 

    return (t_trajectory, x_trajectory)
end 

cpu(data::SingleTrajectoryBatchedOSADataloader) = SingleTrajectoryBatchedOSADataloader(Array(data.data), data.t, data.N, data.N_batch)
gpu(data::SingleTrajectoryBatchedOSADataloader) = SingleTrajectoryBatchedOSADataloader(DeviceArray(data.data), data.t, data.N, data.N_batch)