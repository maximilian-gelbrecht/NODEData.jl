module NODEData


global const cuda_used = Ref(false)

using CUDA

function __init__() # automatically called at runtime to set cuda_used
    cuda_used[] = false
end
DeviceArray(x::AbstractArray) = cuda_used[] ? CuArray(x) : Array(x)

"""
    gpuon()

Manually toggle GPU use on (if available)
"""
function gpuon() # manually toggle GPU use on and off
    cuda_used[] = CUDA.functional()
end

"""
    gpuoff()

Manually toggle GPU use off
"""
function gpuoff()
    cuda_used[] = false
end

abstract type AbstractNODEDataloader{T,U,N} end

include("data.jl")
include("largedata.jl")

export NODEDataloader, LargeNODEDataloader, delete

end
