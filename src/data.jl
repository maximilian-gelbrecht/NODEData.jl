using SciMLBase, EllipsisNotation

"""
    NODEDataloader{T,N} <: AbstractNODEDataloader{T,N}

Struct containing batched data for sequence learning of ODEs. Can be indexed and interated over.

# Inititilized with

    NODEData(sol::SciMLBase.AbstractTimeseriesSolution, N_length::Integer; dt=nothing)

* `sol`: DE solution
* `N_length`: length of each batch
* `dt`: time increment that `sol` is interpolated at. If `nothing` then the `sol.t` is used as the time steps of the data
"""
struct NODEDataloader{T,N} <: AbstractNODEDataloader{T,N}
    data::AbstractArray{T,N}
    t::AbstractArray{T,1}
    N::Integer
    N_length::Integer
end

function NODEDataloader(sol::SciMLBase.AbstractTimeseriesSolution, N_length::Integer; dt=nothing)

    if isnothing(dt)
        data = Array(sol)
        t = sol.t
    else
        t = sol.t[1]:dt:sol.t[end]
        data = zeros(eltype(sol(0.)), size(sol(0.))..., length(t))
        for (i,it) ∈ enumerate(t)
            data[..,i] = sol(it)
        end
    end

    N_t = length(t)
    N = N_t - N_length

    NODEDataloader(togpu(data), togpu(t), N, N_length)
end

function Base.getindex(iter::NODEDataloader{T,N}, i::Integer) where {T,N}
    @assert 0 < i <= iter.N
    return (iter.t[i:i+iter.N_length] ,iter.data[..,i:i+iter.N_length])
end

function Base.iterate(iter::AbstractNODEDataloader, state=1)
    if state>iter.N
        return nothing
    else
        return (iter[state], state+1)
    end
end

Base.length(iter::AbstractNODEDataloader) = iter.N
Base.eltype(iter::AbstractNODEDataloader) = Array{typeof(iter.data),1}

Base.firstindex(iter::AbstractNODEDataloader) = 1
Base.lastindex(iter::AbstractNODEDataloader) = iter.N
Base.show(io::IO,seq::NODEDataloader{T,N}) where {T,N} = print(io, "NODEData{",T,",",N,"} with ",seq.N," batches with length ",seq.N_length)
