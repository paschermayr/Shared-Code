#Comment to https://github.com/invenia/ParameterHandling.jl/pull/39
############################################################################################
using ChainRulesCore
@inline function flatten_array(mat::AbstractArray{R}) where {R<:Real}
    return vec(mat)
end
@inline function fill_array!(buffer::AbstractArray{T}, vec::Union{F, AbstractArray{F}}) where {T<:Real, F<:Real}
    @inbounds for iter in eachindex(vec)
        buffer[iter] = vec[iter]
    end
    return buffer
end
function ChainRulesCore.rrule(::typeof(fill_array!), mat::AbstractMatrix{R}, v::Union{R, AbstractVector{R}}) where {R<:Real}
    # forward pass: Fill Matrix with Vector elements
    L = fill_array!(mat, v)
    # backward pass: Fill Vector with Matrix elements
    pullback_Idx(Δ) = ChainRulesCore.NoTangent(), ChainRulesCore.unthunk(Δ), flatten_array(ChainRulesCore.unthunk(Δ))
    return L, pullback_Idx
end

############################################################################################
function flatten end
flatten(x) = flatten(Float64, true, x)
flatten(strict, x) = flatten(Float64, strict, x)

function _flatten(::Type{T}, strict::Bool, x) where {T<:AbstractFloat}
    _unflatten_Fixed(v) = x
    return T[], _unflatten_Fixed
end

function flatten(::Type{T}, strict::Bool, x::Union{I, Array{I}}) where {T<:AbstractFloat, I<:Integer}
    v = I[]
    unflatten_Integer(v) = x
    return v, unflatten_Integer
end

function flatten(::Type{T}, strict::Bool, x::F) where {T<:AbstractFloat, F<:AbstractFloat}
    v = T[x]
    if strict
        unflatten_to_Real(v) = convert(F, only(v))
        return v, unflatten_to_Real
    else
        unflatten_to_Real_AD(v) = only(v)
        return v, unflatten_to_Real_AD
    end
end

function flatten(::Type{T}, strict::Bool, x::AbstractVector{F}) where {T<:AbstractFloat, F<:AbstractFloat}
    if strict
        buffer = zeros(F, size(x))
        unflatten_to_Vec(v) = fill_array!(buffer, v)
        return Vector{T}(x), unflatten_to_Vec
    else
        return Vector{T}(x), identity
    end
end
function flatten(::Type{T}, strict::Bool, x::AbstractArray{F}) where {T<:AbstractFloat, F<:AbstractFloat}
    x_vec, from_vec = flatten(T, strict, vec(x))
    if strict
        buffer = zeros(F, size(x))
        unflatten_to_Array(v) = fill_array!(buffer, v)
        return x_vec, unflatten_to_Array
    else
        unflatten_to_Array_AD(v) = fill_array!(zeros(eltype(x_vec), size(x)), v)
        return x_vec, unflatten_to_Array_AD
    end
end

function flatten(::Type{T}, strict::Bool, x::AbstractArray) where {T<:AbstractFloat}
    x_vecs_and_backs = map(x) do xᵢ
        flatten(T, strict, xᵢ)
    end
    x_vecs, backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    sz = cumsum( map(length, x_vecs) )
    if strict
        function unflatten_to_AbstractArray(x_vec)
            x_Vec = [backs[n](@view(x_vec[sz[n] - length(x_vecs[n]) + 1:sz[n]])) for n in eachindex(x)]
            return x_Vec
        end
        return reduce(vcat, x_vecs), unflatten_to_AbstractArray
    else
        function unflatten_to_AbstractArray_AD(x_vec)
            x_Vec = [backs[n](x_vec[sz[n] - length(x_vecs[n]) + 1:sz[n]]) for n in eachindex(x)]
            return x_Vec
        end
        return reduce(vcat, x_vecs), unflatten_to_AbstractArray_AD
    end
end

function flatten(::Type{T}, strict::Bool, x::Tuple) where {T<:AbstractFloat}
    x_vecs_and_backs = map(x) do xᵢ
        flatten(T, strict, xᵢ)
    end
    x_vecs, x_backs = first.(x_vecs_and_backs), last.(x_vecs_and_backs)
    lengths = map(length, x_vecs)
    sz = cumsum(lengths)
    if strict
        function unflatten_to_Tuple(v::AbstractVector{<:Real})
            map(x_backs, lengths, sz) do x_back, l, s
                return x_back(@view(v[s - l + 1:s]))
            end
        end
        return reduce(vcat, x_vecs), unflatten_to_Tuple
    else
        function unflatten_to_Tuple_AD(v::AbstractVector{<:Real})
            map(x_backs, lengths, sz) do x_back, l, s
                return x_back(v[s - l + 1:s])
            end
        end
        return reduce(vcat, x_vecs), unflatten_to_Tuple_AD
    end
end

function flatten(::Type{T}, strict::Bool, x::NamedTuple{names}) where {T<:AbstractFloat, names}
    x_vec, unflatten = flatten(T, strict, values(x))
    if strict
        function unflatten_to_NamedTuple(v::AbstractVector{<:Real})
            v_vec_vec = unflatten(v)
            return typeof(x)(v_vec_vec)
        end
        return x_vec, unflatten_to_NamedTuple
    else
        function unflatten_to_NamedTuple_AD(v::AbstractVector{<:Real})
            v_vec_vec = unflatten(v)
            return NamedTuple{names}(v_vec_vec)
        end
        return x_vec, unflatten_to_NamedTuple_AD
    end
end
