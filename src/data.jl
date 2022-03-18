using SciMLBase, EllipsisNotation

"""
    NODEDataloader{T,N} <: AbstractNODEDataloader{T,N}

Struct containing batched data for sequence learning of ODEs. Can be indexed and interated over. Each batch returns a tuple (t, data(t)).

# Initialized with

    NODEData(sol::SciMLBase.AbstractTimeseriesSolution, N_length::Integer; dt=nothing, valid_set=nothing)

* `sol`: DE solution
* `N_length`: length of each batch
* `dt`: time increment that `sol` is interpolated at. If `nothing` then the `sol.t` is used as the time steps of the data
* `valid_set` if valid_set ∈ [0,1] splits the data into a train and valid set with `valid_set` of the share of the data belonging to the valid_set.

    NODEDataloader(data::AbstractArray{T,N}, t::AbstractArray{T,1}, N_length::Integer)

* `data`:: Data that is already in N_dim_1 x ... x N_t format
* `t`: time axis
* `N_length`: length of each batch

"""
struct NODEDataloader{T,U,N} <: AbstractNODEDataloader{T,U,N}
    data::AbstractArray{T,N}
    t::AbstractArray{U,1}
    N::Integer
    N_length::Integer
end

function NODEDataloader(sol::Union{SciMLBase.AbstractTimeseriesSolution, SciMLBase.AbstractDiffEqArray}, N_length::Integer; dt=nothing, valid_set=nothing)

    if isnothing(dt)
        data = DeviceArray(sol)
        t = sol.t
    else
        t = sol.t[1]:eltype(sol)(dt):sol.t[end]
        data = DeviceArray(sol(t))
    end

    if isnothing(valid_set)
        N_t = length(t)
        N = N_t - N_length

        return NODEDataloader(DeviceArray(data), t, N, N_length)
    else
        @assert 0<valid_set<1

        N_t = length(t)
        N_t_valid = Int(floor(valid_set*N_t))
        N_t_train = N_t - N_t_valid

        return NODEDataloader(DeviceArray(data[..,1:N_t_train]), t[1:N_t_train], N_t_train - N_length, N_length), NODEDataloader(DeviceArray(data[..,N_t_train+1:N_t]), t[N_t_train+1:N_t], N_t_valid - N_length, N_length)
    end
end

NODEDataloader(data::AbstractArray{T,N}, t::AbstractArray{U,1}, N_length::Integer) where {T,U,N} = NODEDataloader(DeviceArray(data), t, length(t) - N_length, N_length)
NODEDataloader(data::NODEDataloader, N_length::Integer) = NODEDataloader(data.data, data.t, N_length)

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