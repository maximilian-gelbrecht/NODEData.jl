using SciMLBase, EllipsisNotation

"""
    NODEDataloader{T,N} <: AbstractNODEDataloader{T,N}

Struct containing batched data for sequence learning of ODEs. Can be indexed and interated over. Each batch returns a tuple (t, data(t)).

# Initialized with

    NODEData(sol::SciMLBase.AbstractTimeseriesSolution, N_length::Integer; dt=nothing, valid_set=nothing, GPU=nothing)

* `sol`: DE solution
* `N_length`: length of each batch
* `dt`: time increment that `sol` is interpolated at. If `nothing` then the `sol.t` is used as the time steps of the data
* `valid_set` if valid_set ∈ [0,1] splits the data into a train and valid set with `valid_set` of the share of the data belonging to the valid_set.
* `GPU` if `nothing` the output is automatically chosen to be on GPU/CPU based on `sol`, if `GPU==true` or `GPU==false` the automatic choice is overwritten

    NODEDataloader(data::AbstractArray{T,N}, t::AbstractArray{T,1}, N_length::Integer)

* `data`:: Data that is already in N_dim_1 x ... x N_t format
* `t`: time axis
* `N_length`: length of each batch

"""
struct NODEDataloader{T<:AbstractArray,U<:AbstractVector,N<:Integer} <: AbstractNODEDataloader{T,U,N}
    data::T
    t::U
    N::N
    N_length::N
end

function NODEDataloader(sol::Union{SciMLBase.AbstractTimeseriesSolution, SciMLBase.AbstractDiffEqArray}, N_length::Integer; dt=nothing, GPU::Union{Bool, Nothing}=nothing, kwargs...)

    set_gpu(detect_sol_array_type(sol), GPU)

    if isnothing(dt)
        data = DeviceArray(sol)
        t = Array(sol.t)
    else
        t = sol.t[1]:eltype(sol)(dt):sol.t[end]
        data = DeviceArray(sol(t))
    end

    NODEDataloader(data, t, N_length; GPU=GPU, kwargs...)
end

function NODEDataloader(data::AbstractArray{T,N}, t::AbstractArray{U,1}, N_length::Integer; valid_set=nothing, test_set=nothing, GPU::Union{Bool, Nothing}=nothing) where {T,U,N} 
    @assert size(data)[end] == length(t) "Length of data and t should be equal"
    
    set_gpu(typeof(data) <: CuArray ? true : false, GPU)

    if isnothing(valid_set) 
        return NODEDataloader(DeviceArray(data), Array(t), length(t) - N_length +1 , N_length)
    else 
        @assert 0 <= valid_set < 1 "valid_set should be ∈ [0,1]"
        if !isnothing(test_set)
             @assert 0 <= test_set < 1 "test_set should be ∈ [0,1]"
             @assert test_set + valid_set < 1 "test_set + valid_set shoud be < 1"
        end  

        N_t = length(t)
        N_t_valid = Int(floor(valid_set*N_t))
        N_t_train = N_t - N_t_valid

        if N_t_train - N_length + 1 <= 0 
            @warn "Empty Training set"
        end 

        if  N_t_valid - N_length + 1 <= 0 
            @warn "Empty Valid set, not enough data"
        end

        if isnothing(test_set)
            return NODEDataloader(DeviceArray(data[..,1:N_t_train]), Array(t[1:N_t_train]), N_t_train - N_length + 1, N_length), NODEDataloader(DeviceArray(data[..,N_t_train+1:N_t]), Array(t[N_t_train+1:N_t]), N_t_valid - N_length + 1, N_length)
        else 
            N_t_test = Int(floor(test_set*N_t))
            N_t_train = N_t - N_t_valid - N_t_test

            if  N_t_test - N_length + 1 <= 0 
                @warn "Empty test set, not enough data"
            end

            return NODEDataloader(DeviceArray(data[..,1:N_t_train]), Array(t[1:N_t_train]), N_t_train - N_length + 1, N_length), NODEDataloader(DeviceArray(data[..,N_t_train+1:N_t_train+N_t_valid]), Array(t[N_t_train+1:N_t_train+N_t_valid]), N_t_valid - N_length + 1, N_length), NODEDataloader(DeviceArray(data[..,N_t_train+N_t_valid+1:N_t]), Array(t[N_t_train+N_t_valid+1:N_t]), N_t_test - N_length + 1, N_length)
        end 
    end
end 

NODEDataloader(data::NODEDataloader, N_length::Integer) = NODEDataloader(data.data, data.t, N_length)
remake_dataloader(data::NODEDataloader, N_length::Integer) = NODEDataloader(data, N_length)

function Base.getindex(iter::NODEDataloader{T,U,N}, i::Integer) where {T,U,N}
    @assert 0 < i <= iter.N
    return (iter.t[i:i+iter.N_length-1] ,iter.data[..,i:i+iter.N_length-1])
end

function Base.iterate(iter::AbstractNODEDataloader, state=1)
    if state>iter.N
        return nothing
    else
        return (iter[state], state+1)
    end
end

Base.length(iter::AbstractNODEDataloader) = iter.N
Base.eltype(iter::AbstractNODEDataloader) = eltype(iter.data)

Base.firstindex(iter::AbstractNODEDataloader) = 1
Base.lastindex(iter::AbstractNODEDataloader) = iter.N
Base.show(io::IO,seq::NODEDataloader{T,U,N}) where {T,U,N} = print(io, "NODEData{",T,",",N,"} with ",seq.N," batches with length ",seq.N_length)

"""
    get_trajectory(data::NODEDataloader, N)

Returns a (t, x(t)) tuple of length `N`.
"""
function get_trajectory(data::NODEDataloader, N)
    ind = 1:N 
    (data.t[ind], data.data[..,ind])
end 

cpu(data::NODEDataloader) = NODEDataloader(Array(data.data), data.t, data.N, data.N_length)
gpu(data::NODEDataloader) = NODEDataloader(DeviceArray(data.data), data.t, data.N, data.N_length)

function set_gpu(autodetect_GPU::Bool, manual_GPU::Union{Bool, Nothing}=nothing)
    if !isnothing(manual_GPU)
        GPU = manual_GPU
    else 
        GPU = autodetect_GPU
    end 
    if GPU 
        gpuon()
    else
        gpuoff()
    end 
end 

""" 
    detect_sol_array_type(sol::Union{SciMLBase.AbstractTimeseriesSolution, SciMLBase.AbstractDiffEqArray})

Returns `true` if the solution is on a GPU, `false` if it is not CPU. 
"""
function detect_sol_array_type(sol::Union{SciMLBase.AbstractDiffEqArray, SciMLBase.AbstractTimeseriesSolution})    
    arraytype = typeof(sol.u[1])

    if arraytype <: CuArray 
        return true 
    elseif arraytype <: AbstractArray 
        return false 
    elseif arraytype <: Number 
        if typeof(sol.u) <: CuArray 
            return true 
        elseif typeof(sol.u) <: AbstractArray 
            return false 
        else 
            error("Can't determine array type of solution")
        end 
    else
        error("Can't determine array type of solution")
    end 
end 

