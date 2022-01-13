module NODEData


global const cuda_used = Ref(false)

using CUDA

function __init__() # automatically called at runtime to set cuda_used
    cuda_used[] = CUDA.functional()
end
DeviceArray(x::AbstractArray) = cuda_used[] ? CuArray(x) : Array(x)

abstract type AbstractNODEDataloader{T,U,N} end

include("data.jl")

export NODEDataloader

end
