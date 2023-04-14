# TODO: remove once FillArrays compat for <v1 is dropped
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
size(v::OneHotVector) = (v.len,)
length(v::OneHotVector) = v.len
function getindex(v::OneHotVector{T}, i::Int) where {T}
	i == v.n ? one(T) : zero(T)
end
@static if isdefined(FillArrays, :OneElement)
	const _OneElement = FillArrays.OneElement
else
	const _OneElement = OneHotVector
end
# assume that the basis label starts at zero
function basisfunction(sp, oneindex)
	oneindex >= 0 || throw(ArgumentError("index to set to one must be non-negative, received $oneindex"))
	Fun(sp, _OneElement{Float64}(oneindex, oneindex))
end
