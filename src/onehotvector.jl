struct OneHotVector{T} <: AbstractVector{T}
	n :: Int
	len :: Int

	function OneHotVector{T}(n, len) where {T}
		len >= 0 || throw(ArgumentError("length must be non-negative"))
		0 <= n <= len || throw(ArgumentError("index must be <= length"))
		new{T}(n, len)
	end
end
OneHotVector(n, len = n) = OneHotVector{Float64}(n, len)
Base.size(v::OneHotVector) = (v.len,)
Base.length(v::OneHotVector) = v.len
function Base.getindex(v::OneHotVector{T}, i::Int) where {T}
	i == v.n ? one(T) : zero(T)
end
# assume that the basis label starts at zero
function basisfunction(sp, k)
	k >= 0 || throw(ArgumentError("basis label must be non-negative, received $k"))
	Fun(sp, OneHotVector(k+1))
end
