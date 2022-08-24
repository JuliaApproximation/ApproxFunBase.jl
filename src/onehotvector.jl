export basis

struct OneHotVector{T} <: AbstractVector{T}
	n :: Int
	len :: Int
end
OneHotVector(n, len = n) = OneHotVector{Float64}(n, len)
Base.size(v::OneHotVector) = (v.len,)
Base.length(v::OneHotVector) = v.len
function Base.getindex(v::OneHotVector{T}, i::Int) where {T}
	i == v.n ? one(T) : zero(T)
end
basis(sp, k) = Fun(sp, OneHotVector(k))
