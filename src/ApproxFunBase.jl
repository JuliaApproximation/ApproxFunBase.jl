module ApproxFunBase
using Base: AnyDict
using Base, BlockArrays, BandedMatrices, BlockBandedMatrices, DomainSets,
              IntervalSets, SpecialFunctions, AbstractFFTs, FFTW,
              SpecialFunctions, DSP, DualNumbers, LinearAlgebra, SparseArrays,
              LowRankApprox, FillArrays, InfiniteArrays, InfiniteLinearAlgebra
              # Arpack

import StaticArrays, Calculus
import StaticArrays: SVector, @SArray

import DomainSets: Domain, indomain, UnionDomain, ProductDomain, Point, ∂,
              elements, DifferenceDomain, Interval, ChebyshevInterval, boundary,
              rightendpoint, leftendpoint, dimension, WrappedDomain, VcatDomain,
              component, components, ncomponents

using AbstractFFTs: Plan

import Base: values, convert, getindex, setindex!, *, +, -, ==, <, <=, >, |, !,
              !=, eltype, iterate, /, ^, \,
              transpose, size, tail, broadcast, broadcast!, copyto!, copy,
              to_index, (:), similar, map, vcat, hcat, hvcat, show, summary,
              stride, sum, cumsum, imag, conj, inv, complex, reverse, exp,
              sqrt, abs, abs2, sign, issubset, in, first, last, rand, intersect,
              setdiff, isless, union, angle, join, isnan, isapprox, isempty,
              sort, merge, minimum, maximum, extrema, argmax,
              argmin, findmax, findmin, isfinite,
              zeros, zero, one, promote_rule, repeat, length, resize!, isinf,
              getproperty, findfirst, unsafe_getindex, fld, div,
              eachindex, firstindex, lastindex, isreal,
              OneTo, Array, Vector, Matrix, view, ones, @propagate_inbounds,
              print_array, split, iszero, permutedims, rad2deg, deg2rad

import Base.Broadcast: BroadcastStyle, Broadcasted, AbstractArrayStyle,
              broadcastable, DefaultArrayStyle, broadcasted

import Statistics: mean

import Combinatorics: multiexponents

import LinearAlgebra: BlasInt, BlasFloat, norm, ldiv!, mul!, det, cross,
              qr, qr!, rank, isdiag, istril, istriu, issymmetric,
              Tridiagonal, diagm, diagm_container, factorize,
              nullspace, Hermitian, Symmetric, adjoint, transpose, char_uplo

import SparseArrays: blockdiag

# import Arpack: eigs

# we need to import all special functions to use Calculus.symbolic_derivatives_1arg
# we can't do importall Base as we replace some Base definitions
import SpecialFunctions: sinpi, cospi, airy, besselh,
              sin, cos, cosh, exp2, exp10, log2, log10, csc, acsc, sec,
              asec, cot, acot, sinh, csch, asinh, acsch,
              sech, acosh, asech, tanh, coth, atanh, acoth,
              log1p, lfact, sinc, cosc, beta, lbeta,
              eta, zeta, polygamma, logabsgamma, loggamma,
              abs, sign, log, expm1, tan, abs2, sqrt, angle,
              max, min, cbrt, atan, acos, asin, inv,
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
              bandwidths, _BandedMatrix, BandedMatrix

import BlockArrays: blocksize, block, blockaxes, blockindex
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

import Base: view

import DomainSets: dimension

import IntervalSets: (..), endpoints

const Vec{d,T} = SVector{d,T}

export pad!, pad, chop!, sample,
       complexroots, roots,
       reverseorientation

export .., Interval, ChebyshevInterval, leftendpoint, rightendpoint
export endpoints, cache

import Base: oneto

# assert that the conversion succeeds. This helps with inference as well as sanity
strictconvert(T::Type, x) = convert(T, x)::T

uniontypedvec(A, B) = Union{typeof(A), typeof(B)}[A, B]

convert_vector(v::AbstractVector) = convert(Vector, v)
convert_vector(t::Tuple) = [t...]

promote_eltypeof(As...) = promote_eltypeof(As)
# Avoid mapreduce for common cases, as it often suffers from poor type inference
promote_eltypeof(As::Tuple{Any}) = eltype(As[1])
promote_eltypeof(As::Tuple{Any,Any}) = promote_type(eltype(As[1]), eltype(As[2]))
promote_eltypeof(As::Tuple{Any,Any,Any}) = promote_type(eltype(As[1]), eltype(As[2]), eltype(As[3]))
promote_eltypeof(As::Union{AbstractArray, Tuple}) = mapfoldl(eltype, promote_type, As)

include("LinearAlgebra/LinearAlgebra.jl")
include("Fun.jl")
include("onehotvector.jl")
include("Domains/Domains.jl")
include("Multivariate/Multivariate.jl")
include("Operators/Operator.jl")
include("Caching/caching.jl")
include("PDE/PDE.jl")
include("Spaces/Spaces.jl")
include("hacks.jl")
include("testing.jl")
include("specialfunctions.jl")
include("show.jl")

end #module
