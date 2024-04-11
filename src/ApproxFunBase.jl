module ApproxFunBase

using AbstractFFTs
using BandedMatrices
using BlockArrays
using BlockBandedMatrices
import Calculus
import Combinatorics: multiexponents
using DSP
using DomainSets
using DualNumbers
using FFTW
using FillArrays
using InfiniteArrays
using IntervalSets
using LinearAlgebra
using LowRankMatrices
using SpecialFunctions
using StaticArrays: SVector, @SArray, SArray

import DomainSets: Domain, indomain, UnionDomain, ProductDomain, Point, ∂,
              SetdiffDomain, Interval, ChebyshevInterval, boundary,
              rightendpoint, leftendpoint, dimension, WrappedDomain, VcatDomain,
              component, components, ncomponents, factor, factors, nfactors,
              canonicaldomain

using AbstractFFTs: Plan

import Base: values, convert, getindex, setindex!, *, +, -, ==, <, <=, >, |, !,
              !=, eltype, iterate, /, ^, \,
              transpose, size, ndims, tail, broadcast, broadcast!, copyto!, copy,
              to_index, (:), similar, map, vcat, hcat, hvcat, show, summary,
              stride, sum, cumsum, imag, conj, inv, complex, reverse, exp,
              sqrt, abs, abs2, sign, issubset, in, first, last, rand, intersect,
              setdiff, isless, union, angle, join, isnan, isapprox, isempty,
              sort, merge, minimum, maximum, extrema, argmax,
              argmin, findmax, findmin, isfinite,
              zeros, zero, one, promote_rule, repeat, length, resize!, reshape, isinf,
              getproperty, findfirst, unsafe_getindex, fld, div,
              eachindex, firstindex, lastindex, isreal,
              OneTo, Array, Vector, Matrix, view, ones, @propagate_inbounds,
              print_array, split, iszero, permutedims, rad2deg, deg2rad, checkbounds,
              real, float, view, oneto,
              sinpi, cospi, sin, cos, cosh, exp2, exp10, log2, log10, csc, acsc, sec,
              asec, cot, acot, sinh, csch, asinh, acsch,
              sech, acosh, asech, tanh, coth, atanh, acoth,
              sinc, cosc, log1p, log, expm1, tan,
              max, min, cbrt, atan, acos, asin

import Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle,
              broadcastable, DefaultArrayStyle, broadcasted

import LinearAlgebra: BlasInt, BlasFloat, norm, ldiv!, mul!, det, cross,
              qr, qr!, rank, isdiag, istril, istriu, issymmetric,
              Tridiagonal, diagm, diagm_container, factorize,
              nullspace, Hermitian, Symmetric, adjoint, transpose, char_uplo,
              axpy!, eigvals

# we need to import all special functions to use Calculus.symbolic_derivatives_1arg
# we can't do importall Base as we replace some Base definitions
import SpecialFunctions: airy, besselh,
              lfact, beta, lbeta,
              eta, zeta, polygamma, logabsgamma, loggamma,
              besselj, bessely, besseli, besselk, besselkx,
              hankelh1, hankelh2, hankelh1x, hankelh2x,
              # functions from Calculus.symbolic_derivatives_1arg
              erf, erfinv, erfc, erfcinv, erfi, gamma, lgamma,
              digamma, invdigamma, trigamma,
              airyai, airybi, airyaiprime, airybiprime,
              besselj0, besselj1, bessely0, bessely1,
              erfcx, dawson

import BandedMatrices: bandrange, inbands_setindex!, bandwidth,
              colstart, colstop, colrange, rowstart, rowstop, rowrange,
              bandwidths, _BandedMatrix, BandedMatrix, isbanded

import BlockArrays: blocksize, block, blockaxes, blockindex, blocklengths
import BlockBandedMatrices: blockbandwidth, blockbandwidths, blockcolstop,
              blockcolrange, blockcolstart, blockrowstop, blockrowstart,
              subblockbandwidth, subblockbandwidths, _BlockBandedMatrix,
              _BandedBlockBandedMatrix, BandedBlockBandedMatrix,
              BlockBandedMatrix, isblockbanded, isbandedblockbanded,
              bb_numentries, BlockBandedSizes

import FillArrays: AbstractFill, getindex_value
import LazyArrays: cache, CachedVector, cacheddata
import InfiniteArrays: PosInfinity, InfRanges, AbstractInfUnitRange,
              OneToInf, InfiniteCardinal


# convenience for 1-d block ranges
const BlockRange1 = BlockRange{1,Tuple{UnitRange{Int}}}

import DomainSets: dimension

import IntervalSets: (..), endpoints

export pad!, pad, chop!, sample,
       complexroots, roots,
       reverseorientation

export .., Interval, ChebyshevInterval, leftendpoint, rightendpoint
export endpoints, cache

export normalizedspace


# assert that the conversion succeeds. This helps with inference as well as sanity
strictconvert(T::Type, x) = convert(T, x)::T

uniontypedvec(A, B) = Union{typeof(A), typeof(B)}[A, B]

convert_vector(v::AbstractVector) = convert(Vector, v)
convert_vector(t::Tuple) = [t...]

convert_vector_or_svector(v::AbstractVector) = convert(Vector, v)
convert_vector_or_svector(t::Tuple) = SVector(t)

convert_vector_or_svector_promotetypes(v::AbstractVector) = convert_vector(v)
_uniontypes_svector(t) = SVector{length(t), mapfoldl(typeof, (x,y)->Union{x,y}, t)}(t)
convert_vector_or_svector_promotetypes(t::NTuple{2,Any}) = _uniontypes_svector(t)
convert_vector_or_svector_promotetypes(t::NTuple{3,Any}) = _uniontypes_svector(t)
convert_vector_or_svector_promotetypes(t::NTuple{4,Any}) = _uniontypes_svector(t)
convert_vector_or_svector_promotetypes(t::Tuple) = SVector{length(t), mapreduce(typeof, typejoin, t)}(t)

promote_eltypeof(As...) = promote_eltypeof(As)
# Avoid mapreduce for common cases, as it often suffers from poor type inference
promote_eltypeof(As::Tuple{Any}) = eltype(As[1])
promote_eltypeof(As::Tuple{Any,Any}) = promote_type(eltype(As[1]), eltype(As[2]))
promote_eltypeof(As::Tuple{Any,Any,Any}) = promote_type(eltype(As[1]), eltype(As[2]), eltype(As[3]))
promote_eltypeof(As::Union{AbstractArray, Tuple}) = mapfoldl(eltype, promote_type, As)

assert_integer(::Integer) = nothing
function assert_integer(k::Number)
    @assert isinteger(k) "order must be an integer"
    return nothing
end

function _IteratorSize(::Type{T}) where {T<:Tuple}
    s = ntuple(i-> Base.IteratorSize(fieldtype(T, i)), fieldcount(T))
    any(x -> x isa Base.IsInfinite, s) ? Base.IsInfinite() : Base.HasLength()
end

include("LinearAlgebra/LinearAlgebra.jl")
include("Fun.jl")
include("Domains/Domains.jl")
include("Multivariate/Multivariate.jl")
include("Operators/Operator.jl")
include("Caching/caching.jl")
include("PDE/PDE.jl")
include("Spaces/Spaces.jl")
include("eigen.jl")
include("hacks.jl")
include("specialfunctions.jl")
include("show.jl")

if !isdefined(Base, :get_extension)
    include("../ext/ApproxFunBaseSparseArraysExt.jl")
    include("../ext/ApproxFunBaseStatisticsExt.jl")
end

end #module
