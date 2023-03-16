struct OneHotVector{T} <: AbstractVector{T}
	n :: Int
	len :: Int

	function OneHotVector{T}(n, len) where {T}
		len >= 0 || throw(ArgumentError("length must be non-negative"))
		0 <= n <= len || throw(ArgumentError("index must be <= length"))
		new{T}(n, len)
	end
end
OneHotVector{T}(v::OneHotVector) where {T} = OneHotVector{T}(v.n, v.len)
OneHotVector(n, len = n) = OneHotVector{Float64}(n, len)
Base.size(v::OneHotVector) = (v.len,)
Base.length(v::OneHotVector) = v.len
function Base.getindex(v::OneHotVector{T}, i::Int) where {T}
	i == v.n ? one(T) : zero(T)
end
# assume that the basis label starts at zero
function basisfunction(sp, oneindex)
	oneindex >= 0 || throw(ArgumentError("index to set to one must be non-negative, received $oneindex"))
	Fun(sp, OneHotVector(oneindex))
end
