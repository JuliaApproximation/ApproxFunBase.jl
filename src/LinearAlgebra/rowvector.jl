# This file is based on rowvector.jl in Julia. License is MIT: https://julialang.org/license
# The motivation for this file is to allow RowVector which doesn't transpose the entries

import Base: convert, similar, length, size, axes, IndexStyle,
            IndexLinear, @propagate_inbounds, getindex, setindex!,
            broadcast, hcat, typed_hcat, map, parent

"""
    RowVector(vector)

A lazy-view wrapper of an `AbstractVector`, which turns a length-`n` vector into a `1×n`
shaped row vector and represents the transpose of a vector (although unlike `transpose`,
the elements are *not* transposed recursively).

By convention, a vector can be multiplied by a matrix on its left (`A * v`) whereas a row
vector can be multiplied by a matrix on its right (such that `RowVector(v) * A = RowVector(transpose(A) * v)`). It
differs from a `1×n`-sized matrix by the facts that its transpose returns a vector and the
inner product `RowVector(v1) * v2` returns a scalar, but will otherwise behave similarly.
"""
struct RowVector{T,V<:AbstractVector} <: AbstractMatrix{T}
    vec::V
    function RowVector{T,V}(v::V) where V<:AbstractVector where T
        new(v)
    end
end

# Constructors that take a vector
@inline RowVector(vec::AbstractVector{T}) where {T} = RowVector{T,typeof(vec)}(vec)
@inline RowVector{T}(vec::AbstractVector{T}) where {T} = RowVector{T,typeof(vec)}(vec)

# Constructors that take a size and default to Array
@inline RowVector{T}(n::Int) where {T} = RowVector{T}(Vector{T}(undef,n))
@inline RowVector{T}(n1::Int, n2::Int) where {T} = n1 == 1 ?
    RowVector{T}(n2) :
    error("RowVector expects 1×N size, got ($n1,$n2)")
@inline RowVector{T}(n::Tuple{Int}) where {T} = RowVector{T}(n[1])
@inline RowVector{T}(n::Tuple{Int,Int}) where {T} = n[1] == 1 ?
    RowVector{T}(n[2]) :
    error("RowVector expects 1×N size, got $n")

# Conversion of underlying storage
convert(::Type{RowVector{T,V}}, rowvec::RowVector) where {T,V<:AbstractVector} =
    RowVector{T,V}(strictconvert(V,rowvec.vec))

# similar tries to maintain the RowVector wrapper and the parent type
@inline similar(rowvec::RowVector) = RowVector(similar(parent(rowvec)))
@inline similar(rowvec::RowVector, ::Type{T}) where {T} = RowVector(similar(parent(rowvec), T))

# Resizing similar currently loses its RowVector property.
@inline similar(rowvec::RowVector, ::Type{T}, dims::Dims{N}) where {T,N} = similar(parent(rowvec), T, dims)

parent(rowvec::RowVector) = rowvec.vec

# AbstractArray interface
@inline length(rowvec::RowVector) =  length(rowvec.vec)
@inline size(rowvec::RowVector) = (1, length(rowvec.vec))
@inline axes(rowvec::RowVector) = (Base.OneTo(1), axes(rowvec.vec, 1))
IndexStyle(::RowVector) = IndexLinear()
IndexStyle(::Type{<:RowVector}) = IndexLinear()

@propagate_inbounds getindex(rowvec::RowVector, i::Int) = rowvec.vec[i]
@propagate_inbounds setindex!(rowvec::RowVector, v, i::Int) = setindex!(rowvec.vec, v, i)

# helper function for below
to_vec(r::RowVector) = parent(r)
to_vec(x) = x
@inline to_vecs(rowvecs...) = map(to_vec, rowvecs)
# map: Preserve the RowVector by un-wrapping and re-wrapping, but note that `f`
# expects to operate within the transposed domain, so to_vec transposes the elements
@inline map(f, rowvecs::RowVector...) = RowVector(map(f, to_vecs(rowvecs...)...))

# broacast (other combinations default to higher-dimensional array)
@inline broadcast(f, rowvecs::Union{Number,RowVector}...) =
    RowVector(broadcast(f, to_vecs(rowvecs...)...))

# Horizontal concatenation #

@inline hcat(X::RowVector...) = RowVector(mapreduce(parent, vcat, X))
@inline hcat(X::Union{RowVector,Number}...) = RowVector(mapreduce(to_vec, vcat, X))

@inline typed_hcat(::Type{T}, X::RowVector...) where {T} =
    RowVector(Base.typed_vcat(T, to_vecs(X...)...))
@inline typed_hcat(::Type{T}, X::Union{RowVector,Number}...) where {T} =
    RowVector(Base.typed_vcat(T, to_vecs(X...)...))

# Multiplication #

# inner product -> dot product specializations
@inline *(rowvec::RowVector{T}, vec::AbstractVector{T}) where {T<:Real} = dotu(parent(rowvec), vec)

# Generic behavior
@inline function *(rowvec::RowVector, vec::AbstractVector)
    if length(rowvec) != length(vec)
        throw(DimensionMismatch("A has dimensions $(size(rowvec)) but B has dimensions $(size(vec))"))
    end
    sum(@inbounds(rowvec[i]*vec[i]) for i in eachindex(rowvec, vec))
end
@inline *(rowvec::RowVector, mat::AbstractMatrix) = RowVector(transpose(mat) * parent(rowvec))
*(::RowVector, ::RowVector) = throw(DimensionMismatch("Cannot multiply two transposed vectors"))
@inline *(vec::AbstractVector, rowvec::RowVector) = vec .* rowvec



## Removed A_* overrides for now
