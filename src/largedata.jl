# This contains code that deals with data that has to be saved/loaded constantly because it is too large to hold in the RAM all at once 
using JLD2 


"""
    LargeNODEDataloader(sol, N_batch, N_length, name, base_path=""; dt=nothing, valid_set=nothing)

Impelements an array of NODEDataloader that are too large for the RAM and are thus saved to the hard drive (temporally). They can be loaded by just indexing the 
"""
struct LargeNODEDataloader
    name 
    base_path 
    N 
end

function LargeNODEDataloader(sol, N_batch, N_length, name, base_path=""; dt=nothing, valid_set=nothing)

    t = sol.t[1]:eltype(sol)(dt):sol.t[end]

    N_t = length(t)
    N_t_batch = Int(floor(N_t / N_batch))

    try 
        mkdir("temp-data")
    catch 
        nothing 
    end 

    for i=1:N_batch

        train = NODEDataloader(sol(t[(i-1)*N_t_batch+1:i*N_t_batch]), N_length)

        if !(isnothing(valid_set)) && i==N_batch
            global valid = train 
        else 
            save_name = string(base_path, "temp-data/",name, "-",i,".jld2")
            @save save_name train
        end

    end

    N_batch = !(isnothing(valid_set)) ? N_batch - 1 : N_batch

    if !(isnothing(valid_set))
        return LargeNODEDataloader(name, base_path, N_batch), valid 
    else 
        return LargeNODEDataloader(name, base_path, N_batch)
    end
end

function Base.getindex(iter::LargeNODEDataloader, i::Integer) where {T,U,N}
    @assert 0 < i <= iter.N

    save_name = string(iter.base_path, "temp-data/",iter.name, "-",i,".jld2")

    @load save_name train
    return train
end

function Base.iterate(iter::LargeNODEDataloader, state=1)
    if state>iter.N
        return nothing
    else
        return (iter[state], state+1)
    end
end

Base.length(iter::LargeNODEDataloader) = iter.N
Base.eltype(iter::LargeNODEDataloader) = eltype(iter.data)

Base.firstindex(iter::LargeNODEDataloader) = 1
Base.lastindex(iter::LargeNODEDataloader) = iter.N
Base.show(io::IO,seq::LargeNODEDataloader) = print(io, "LargeNODEDataloader with files saved at ",seq.basename,"/tmp-data")

delete(data::LargeNODEDataloader) = rm(string(data.base_path,"temp-data"), recursive=true)