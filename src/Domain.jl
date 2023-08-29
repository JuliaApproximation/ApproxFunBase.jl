export Domain, SegmentDomain, tocanonical, fromcanonical, fromcanonicalD, ∂
export isambiguous, arclength
export components, component, ncomponents

export 𝑪


# add indexing for all spaces, not just DirectSumSpace
# mimicking scalar vs vector

# prectype gives the precision, including for SVector
prectype(::Type{D}) where {D<:Domain} = float(eltype(eltype(D)))
prectype(d::Domain) = prectype(typeof(d))

#TODO: bivariate AnyDomain
struct AnyDomain <: Domain{UnsetNumber} end
struct EmptyDomain <: Domain{Nothing} end

==(::AnyDomain, ::AnyDomain) = true
==(::EmptyDomain, ::EmptyDomain) = true

const AnyOrEmptyDomain = Union{AnyDomain, EmptyDomain}

isambiguous(::AnyDomain) = true
dimension(::AnyDomain) = 1
dimension(::EmptyDomain) = 0

complexlength(::AnyDomain) = NaN
arclength(::AnyDomain) = NaN
arclength(::EmptyDomain) = false
arclength(::DomainSets.EmptySpace) = false

isempty(::AnyDomain) = false

reverseorientation(a::AnyOrEmptyDomain) = a

canonicaldomain(a::AnyOrEmptyDomain) = a

indomain(x::Domain,::EmptyDomain) = false

convert(::Type{Domain{T}}, ::AnyDomain) where T = Domain(T)

# empty domain should have priority over over any domain
_promoteanyemptydomain(::AnyDomain, ::AnyDomain) = AnyDomain()
_promoteanyemptydomain(::AnyOrEmptyDomain, ::AnyOrEmptyDomain) = EmptyDomain()

union(a::AnyOrEmptyDomain, b::AnyOrEmptyDomain) = _promoteanyemptydomain(a, b)
union(::AnyOrEmptyDomain, a::Domain) = a
union(a::Domain, ::AnyOrEmptyDomain) = a

##General routines
isempty(::EmptyDomain) = true


## Interval DomainSets

abstract type SegmentDomain{T} <: Domain{T} end
abstract type AbstractSegment{T} <: SegmentDomain{T} end
const IntervalOrSegment{T} = Union{AbstractInterval{T}, AbstractSegment{T}}
const IntervalOrSegmentDomain{T} = Union{AbstractInterval{T}, SegmentDomain{T}}

canonicaldomain(d::IntervalOrSegmentDomain) = ChebyshevInterval{real(prectype(d))}()

domainscompatible(a,b) = domainscompatible(domain(a),domain(b))
_isapprox(a, b) = isapprox(a,b)
function _isapprox(a::Interval, b::Interval)
    # hacky compat for DomainSets v0.6,
    # where comparing open and closed intevals throws an error
    # However, we don't need to compare at all if the endpoints differ
    # this introduces a duplicate check, but that's probably ok
    isapprox(leftendpoint(a), leftendpoint(b), atol=100eps(prectype(a))) &&
    isapprox(rightendpoint(a), rightendpoint(b), atol=100eps(prectype(a))) &&
    isapprox(a, b)
end
function domainscompatible(a::Domain, b::Domain)
    isambiguous(a) || isambiguous(b) || _isapprox(a, b)
end

domainscompatible(::ChebyshevInterval, ::ChebyshevInterval) = true

##TODO: Should fromcanonical be fromcanonical!?

#TODO consider moving these
leftendpoint(d::IntervalOrSegmentDomain{T}) where {T} = fromcanonical(d,-one(eltype(T)))
rightendpoint(d::IntervalOrSegmentDomain{T}) where {T} = fromcanonical(d,one(eltype(T)))

indomain(x,::AnyDomain) = true
function indomain(x,d::SegmentDomain)
    T=float(real(prectype(d)))
    y=tocanonical(d,x)
    ry=real(y)
    iy=imag(y)
    sc=norm(fromcanonicalD(d,ry<-1 ? -one(ry) : (ry>1 ? one(ry) : ry)))  # scale based on stretch of map on projection to interal
    dy=fromcanonical(d,y)
    # TODO: use isapprox once keywords are fast
    ((isinf(norm(dy)) && isinf(norm(x))) ||  norm(dy-x) ≤ 1000eps(T)*max(norm(x),1)) &&
        -one(T)-100eps(T)/sc ≤ ry ≤ one(T)+100eps(T)/sc &&
        -100eps(T)/sc ≤ iy ≤ 100eps(T)/sc
end

issubcomponent(a::Domain,b::Domain) = a in components(b)


##### canoncial
"""
    canonicaldomain(d)

returns a domain which we map to for operations. For example,
the canonical domain for an interval [a,b] is [-1,1]
"""
function canonicaldomain end


"""
    tocanonical(d, x)

maps the point `x` in `d` to a point in `canonical(d,x)`
"""
function tocanonical end

const 𝑪 = tocanonical

## conveninece routines

ones(d::Domain) = ones(prectype(d),Space(d))
zeros(d::Domain) = zeros(prectype(d),Space(d))


function commondomain(P::AbstractVector)
    ret = AnyDomain()

    for op in P
        d = domain(op)
        @assert ret == AnyDomain() || d == AnyDomain() || ret == d

        if d != AnyDomain()
            ret = d
        end
    end

    ret
end

commondomain(P::AbstractVector,g::AbstractArray{T}) where {T<:Number} = commondomain(P)
commondomain(P::AbstractVector,g) = commondomain([P;g])


domain(::Number) = AnyDomain()


## rand


rand(d::IntervalOrSegmentDomain,k...) = fromcanonical.(Ref(d),2rand(k...)-1)


checkpoints(d::IntervalOrSegmentDomain) = fromcanonical.(Ref(d),SVector{3,Float64}(-0.823972,0.01,0.3273484))

## boundary

boundary(d::SegmentDomain) = [leftendpoint(d),rightendpoint(d)] #TODO: Points domain





## map domains
# we auto vectorize arguments
tocanonical(d::Domain,x,y,z...) = tocanonical(d,SVector(x,y,z...))
fromcanonical(d::Domain,x,y,z...) = fromcanonical(d,SVector(x,y,z...))


mappoint(d1::Domain,d2::Domain,x...) = fromcanonical(d2,tocanonical(d1,x...))
invfromcanonicalD(d::Domain,x...) = 1/fromcanonicalD(d,x...)



## domains in higher dimensions


## sorting
# we sort spaces lexigraphically by default

for OP in (:<,:(<=),:(isless))
    @eval $OP(a::Domain,b::Domain)=$OP(string(a),string(b))
end


## Other special domains

struct PositiveIntegers <: Domain{Int} end
struct Integers <: Domain{Int} end

const ℕ = PositiveIntegers()
const ℤ = Integers()

==(::PositiveIntegers, ::PositiveIntegers) = true
==(::Integers, ::Integers) = true
