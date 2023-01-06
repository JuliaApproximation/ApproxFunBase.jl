export Space, domainspace, rangespace, maxspace,Space,conversion_type, transform,
            itransform, transform!, itransform!, SequenceSpace, ConstantSpace



# Space maps the Domain to the type R
# For example, we have
#   Chebyshev{Interval{Float64}} <: Space{Interval{Float64},Float64}
#   Laurent{PeriodicSegment{Float64}} <: Space{PeriodicSegment{Float64},ComplexF64}
#   Fourier{Circle{ComplexF64}} <: Space{Circle{ComplexF64},Float64}
# Note for now Space doesn't contain any information about the coefficients

abstract type Space{D,R} end



const RealSpace = Space{D,R} where {D,R<:Real}
const ComplexSpace = Space{D,R} where {D,R<:Complex}
const UnivariateSpace = Space{D,R} where {D<:Domain{<:Number},R}
const BivariateSpace = Space{D,R}  where {D<:EuclideanDomain{2},R}
const RealUnivariateSpace = RealSpace{D,R} where {D<:Domain{<:Number},R<:Real}




eltype(S::Space{T}) where {T} = error("Eltype has been changed to domaintype, rangetype or prectype")
eltype(::Type{Space{D,R}}) where {D,R} = error("Eltype has been changed to domaintype, rangetype or prectype")

domaintype(::Space{D,R}) where {D,R} = D
domaintype(::Type{Space{D,R}}) where {D,R} = D
domaintype(::Type{FT}) where {FT<:Space} = domaintype(supertype(FT))
rangetype(::Space{D,R}) where {D,R} = R
rangetype(::Type{Space{D,R}}) where {D,R} = R
rangetype(::Type{FT}) where {FT<:Space} = rangetype(supertype(FT))

domaindimension(sp::Space) = dimension(domain(sp))
"""
    dimension(s::Space)

Return the dimension of `s`, which is the maximum number of coefficients.
"""
dimension(::Space) = ℵ₀  # We assume infinite-dimensional spaces


# add indexing for all spaces, not just DirectSumSpace
# mimicking scalar vs vector



#supports broadcasting, overloaded for ArraySpace
size(::Space) = ()

transpose(sp::Space) = sp  # default no-op


# the default is all spaces have one-coefficient blocks
blocklengths(S::Space) = Ones{Int}(dimension(S))
blocksize(S::Space) = (length(blocklengths(S)),)
blockaxes(S::Space) = (Block.(oneto(length(blocklengths(S)))),)
function blockaxes(A::Space, d)
    @assert d == 1
    blockaxes(A)[1]
end

block(S::Space,k) = Block(k)

Space(s::Space) = s



abstract type AmbiguousSpace <: Space{AnyDomain,UnsetNumber} end
domain(::AmbiguousSpace) = AnyDomain()


function setdomain(sp::Space{D},d::D) where D<:Domain
    S = typeof(sp)
    @assert length(fieldnames(S))==1
    S(d)
end

# function setdomain(sp::Space,d::Domain)
#     S=typeof(sp)
#     @assert length(fieldnames(S))==1
#     # the domain is not compatible, but maybe we c
#     # can drop the space depence.  For example,
#     # CosSpace{Circle{Float64}} -> CosSpace
#     eval(Meta.parse(string(S.name.module)*"."*string(S.name)))(d)
# end

setcanonicaldomain(s) = setdomain(s,canonicaldomain(s))

reverseorientation(S::Space) = setdomain(S,reverseorientation(domain(S)))


# UnsetSpace dictates that an operator is not defined until
#   its domainspace is promoted
# NoSpace is used to indicate no space exists for, e.g.,
# conversion_type

struct UnsetSpace <: AmbiguousSpace end
struct NoSpace <: AmbiguousSpace end

isambiguous(_) = false
isambiguous(::Type{UnsetNumber}) = true
isambiguous(::Type{Array{T}}) where {T} = isambiguous(T)
isambiguous(sp::Space) = isambiguous(rangetype(sp))


#TODO: should it default to canonicalspace?
"""
    points(s::Space,n::Integer)

Return a grid of approximately `n` points, for which a transform exists
from values at the grid to coefficients in the space `s`.

# Examples
```jldoctest
julia> chebypts(n) = [cos((2i+1)pi/2n) for i in 0:n-1];

julia> points(Chebyshev(), 4) ≈ chebypts(4)
true
```
"""
points(d::Space,n) = points(domain(d),n)
points(d::Space) = points(d, dimension(d))

"""
    canonicalspace(s::Space)

Return a space that is used as a default to implement missing functionality,
e.g., evaluation.
Implement a [`Conversion`](@ref) operator or override [`coefficients`](@ref) to support this.

# Examples
```jldoctest
julia> ApproxFunBase.canonicalspace(NormalizedChebyshev())
Chebyshev()
```
"""
canonicalspace(T) = T

canonicaldomain(S::Space) = canonicaldomain(domain(S))

# Check whether spaces are the same, override when you need to check parameters
# This is used in place of == to support AnyDomain
spacescompatible(f::D,g::D) where D<:Space = error("Override spacescompatible for "*string(D))
spacescompatible(::UnsetSpace,::UnsetSpace) = true
spacescompatible(::NoSpace,::NoSpace) = true
"""
    spacescompatible(A::Space, B::Space)

Specifies equality of spaces while also supporting `AnyDomain`.
"""
spacescompatible(f,g) = false
==(A::Space,B::Space) = spacescompatible(A,B) && domain(A) == domain(B)
spacesequal(A::Space,B::Space) = A==B

pointscompatible(f,g) = spacescompatible(f,g)

# check a list of spaces for compatibility
for OP in (:spacescompatible,:domainscompatible,:spacesequal),TYP in (:AbstractArray,:Tuple)
    @eval function $OP(v::$TYP)
        for k=1:length(v)-1
            if !$OP(v[k],v[k+1])
                return false
            end
        end
        true
    end
end



domain(A::Space) = A.domain # assume it has a field domain



for op in (:tocanonical,:fromcanonical,:tocanonicalD,:fromcanonicalD,:invfromcanonicalD)
    @eval ($op)(sp::Space,x...)=$op(domain(sp),x...)
end

_domain(s::Space) = domain(s)
_domain(s) = s
mappoint(a, b, x) = mappoint(map(_domain, (a, b, x))...)

_conversion_rule(a, b) = spacescompatible(a, b) ? a : NoSpace()

for FUNC in (:conversion_rule,:maxspace_rule,:union_rule)
    @eval $FUNC(a, b) = _conversion_rule(a, b)
end

pick_maybe_nonambiguous_space(a::Space, b...) = a
pick_maybe_nonambiguous_space(a::AmbiguousSpace, b) = pick_maybe_nonambiguous_space(b)
pick_maybe_nonambiguous_space(a::NoSpace, b...) = a

"""
    conversion_type(a::Space, b::Space)

Return a `Space` that has a banded [`Conversion`](@ref) operator to both `a` and `b`.
Override `ApproxFun.conversion_rule` when adding new `Conversion` operators.

See also [`maxspace`](@ref)
"""
function conversion_type(a, b)
    if a isa UnsetSpace || b isa UnsetSpace
        return pick_maybe_nonambiguous_space(a, b)
    end
    if spacescompatible(a,b)
        a
    elseif !domainscompatible(a,b)
        NoSpace()  # this avoids having to check eachtime
    else
        cr=conversion_rule(a,b)
        cr==NoSpace() ? conversion_rule(b,a) : cr
    end
end






"""
    maxspace(a::Space, b::Space)

Return a space that has a banded conversion operator FROM `a` and `b`

See also [`conversion_type`](@ref)
"""
maxspace(a,b) = NoSpace()  # TODO: this fixes weird bug with Nothing
function maxspace(a::Space, b::Space)
    if a isa UnsetSpace || b isa UnsetSpace
        return pick_maybe_nonambiguous_space(a, b)
    end

    if spacescompatible(a,b)
        return a
    elseif !domainscompatible(a,b)
        return NoSpace()  # this avoids having to check eachtime
    end



    cr=maxspace_rule(a,b)
    if !isa(cr,NoSpace)
        return cr
    end

    cr=maxspace_rule(b,a)
    if !isa(cr,NoSpace)
        return cr
    end

    cr=conversion_type(a,b)
    if cr==a
        return b
    elseif cr ==b
        return a
    end

    # check if its banded through canonicalspace
    cspa=canonicalspace(a)
    if spacescompatible(cspa,b)
        # we can't call maxspace(cspa,a)
        # maxspace/conversion_type should be implemented for canonicalspace
        error("Override conversion_type or maxspace for "*string(a)*" and "*string(b))
    end
    if cspa != a && maxspace(cspa,a)==cspa
        return maxspace(b,cspa)
    end

    cspb=canonicalspace(b)
    if spacescompatible(cspb,a)
        # we can't call maxspace(cspb,b)
        error("Override conversion_type or maxspace for "*string(a)*" and "*string(b))
    end
    if cspb !=b && maxspace(cspb,b)==cspb
        return maxspace(a,cspb)
    end

    NoSpace()
end




# union combines two spaces
# this is used primarily for addition of two funs
# that may be incompatible
union(a::AmbiguousSpace, b::AmbiguousSpace) = b

function union_by_union_rule(@nospecialize(a::Space), @nospecialize(b::Space))
    if a isa AmbiguousSpace || b isa AmbiguousSpace
        return pick_maybe_nonambiguous_space(a, b)
    end
    if spacescompatible(a,b)
        if isambiguous(domain(a))
            return b
        else
            return a
        end
    end

    cr = union_rule(a,b)
    cr isa NoSpace || return cr

    union_rule(b,a)
end

function union(@nospecialize(a::Space), @nospecialize(b::Space))
    cr = union_by_union_rule(a,b)
    cr isa NoSpace || return cr

    cspa=canonicalspace(a)
    cspb=canonicalspace(b)
    if cspa!=a || cspb!=b
        crc = union_by_union_rule(cspa,cspb)
        crc isa NoSpace || return crc
    end

    cr2=maxspace(a,b)  #Max space since we can convert both to it
    cr2 isa NoSpace || return cr2

    a ⊕ b
end

union(a::Space, bs::Space...) = foldl(union, bs, init = a)

"""
    hasconversion(a,b)

Test whether a banded `Conversion` operator exists.
"""
hasconversion(a,b) = maxspace(a,b) == b


"""
    isconvertible(a::Space, b::Space)

Test whether coefficients may be converted from `a` to `b` through a banded `Conversion` operator.
"""
isconvertible(a,b) = a == b || hasconversion(a,b)

## Conversion routines
#       coefficients(v::AbstractVector,a,b)
# converts from space a to space b
#       coefficients(v::Fun,a)
# is equivalent to coefficients(v.coefficients,v.space,a)
#       coefficients(v::AbstractVector,a,b,c)
# uses an intermediate space b

coefficients(f,sp1,sp2,sp3) = coefficients(coefficients(f,sp1,sp2),sp2,sp3)
coefficients!(f,sp1,sp2,sp3) = coefficients!(coefficients!(f,sp1,sp2),sp2,sp3)

coefficients(f::AbstractVector,::Type{T1},::Type{T2}) where {T1<:Space,T2<:Space} =
    coefficients(f,T1(),T2())
coefficients(f::AbstractVector,::Type{T1},sp2::Space) where {T1<:Space} = coefficients(f,T1(),sp2)
coefficients(f::AbstractVector,sp1::Space,::Type{T2}) where {T2<:Space} = coefficients(f,sp1,T2())

## coefficients defaults to calling Conversion, otherwise it tries to pipe through Chebyshev

_mul_coefficients!!(inplace::Val{true}) = mul_coefficients!
_mul_coefficients!!(inplace::Val{false}) = mul_coefficients
_ldiv_coefficients!!(inplace::Val{true}) = ldiv_coefficients!
_ldiv_coefficients!!(inplace::Val{false}) = ldiv_coefficients

_Fun(v::AbstractVector, sp) = Fun(sp, v)
_Fun(v, sp) = Fun(v, sp)
_maybeconvert(inplace::Val{true}, f, v) = v
_maybeconvert(inplace::Val{false}, f::AbstractVector, v) = strictconvert(Vector{float(eltype(f))}, v)
function defaultcoefficients(f,a,b,inplace = Val(false))
    ct=conversion_type(a,b) # gives a space that has a banded conversion to both a and b

    x = if spacescompatible(a,b)
        f
    elseif hasconversion(a,b)
        _mul_coefficients!!(inplace)(Conversion(a,b),f)
    elseif hasconversion(b,a)
        _ldiv_coefficients!!(inplace)(Conversion(b,a),f)
    else
        csp=canonicalspace(a)

        if spacescompatible(a,csp)# a is csp, so try b
            csp=canonicalspace(b)
        end
        if spacescompatible(a,csp) || spacescompatible(b,csp)
            # b is csp too, so we are stuck, try Fun constructor
            # This only works for the out-of-place version as of now
            _coefficients!!(inplace)(default_Fun(_Fun(f,a),b))
        else
            _coefficients!!(inplace)(f,a,csp,b)
        end
    end
    _maybeconvert(inplace, f, x)
end

"""
    coefficients(cfs::AbstractVector, fromspace::Space, tospace::Space) -> Vector

Convert coefficients in `fromspace` to coefficients in `tospace`

# Examples
```jldoctest
julia> f = Fun(x->(3x^2-1)/2);

julia> coefficients(f, Chebyshev(), Legendre()) ≈ [0,0,1]
true

julia> g = Fun(x->(3x^2-1)/2, Legendre());

julia> coefficients(f, Chebyshev(), Legendre()) ≈ coefficients(g)
true
```
"""
coefficients(f,a,b) = defaultcoefficients(f,a,b)
coefficients!(f,a,b) = defaultcoefficients(f,a,b,Val(true))







## rand
# checkpoints is used to give a list of points to double check
# the expansion
rand(d::Space,k...) = rand(domain(d),k...)
checkpoints(d::Space) = checkpoints(domain(d))



## default transforms
abstract type AbstractTransformPlan{T} <: Plan{T} end

space(P::AbstractTransformPlan) = P.space

# These plans are use to wrap another plan
for Typ in (:TransformPlan,:ITransformPlan)
    @eval begin
        struct $Typ{T,SP,inplace,PL} <: AbstractTransformPlan{T}
            space::SP
            plan::PL
        end
        $Typ(space,plan,::Type{Val{inplace}}) where {inplace} =
            $Typ{eltype(plan),typeof(space),inplace,typeof(plan)}(space,plan)
        # *(P::$Typ, x::AbstractArray) = P.plan*x
    end
end



for Typ in (:CanonicalTransformPlan,:ICanonicalTransformPlan)
    @eval begin
        struct $Typ{T,SP,PL,CSP,inplace} <: AbstractTransformPlan{T}
            space::SP
            plan::PL
            canonicalspace::CSP
        end
        $Typ(space,plan,csp) = $Typ(space,plan,csp,Val(false))
        $Typ(space,plan,csp,ip::Val{inplace}) where {inplace} =
            $Typ{eltype(plan),typeof(space),typeof(plan),typeof(csp),inplace}(space,plan,csp)
    end
end
inplace(::CanonicalTransformPlan{<:Any,<:Any,<:Any,<:Any,IP}) where {IP} = IP
inplace(::ICanonicalTransformPlan{<:Any,<:Any,<:Any,<:Any,IP}) where {IP} = IP

# Canonical plan uses coefficients
function checkcanonicalspace(sp)
    csp = canonicalspace(sp)
    sp == csp && error("Override for $sp")
    csp
end
_plan_transform!!(::Val{true}) = plan_transform!
_plan_transform!!(::Val{false}) = plan_transform
function CanonicalTransformPlan(space, v, inplace::Val = Val(false))
    csp = checkcanonicalspace(space)
    CanonicalTransformPlan(space, _plan_transform!!(inplace)(csp,v), csp, inplace)
end
plan_transform(sp::Space,vals) = CanonicalTransformPlan(sp, vals, Val(false))
plan_transform!(sp::Space,vals) = CanonicalTransformPlan(sp, vals, Val(true))

_plan_itransform!!(::Val{true}) = plan_itransform!
_plan_itransform!!(::Val{false}) = plan_itransform
function ICanonicalTransformPlan(space, v, ip::Val{inplace} = Val(false)) where {inplace}
    csp = checkcanonicalspace(space)
    cfs = inplace ? v : coefficients(v,space,csp)
    ICanonicalTransformPlan(space, _plan_itransform!!(ip)(csp,cfs), csp, ip)
end
plan_itransform(sp::Space,v) = ICanonicalTransformPlan(sp, v, Val(false))
plan_itransform!(sp::Space,v) = ICanonicalTransformPlan(sp, v, Val(true))

# transform converts from values at points(S,n) to coefficients
# itransform converts from coefficients to values at points(S,n)

"""
    transform(s::Space, vals)

Transform values on the grid specified by `points(s,length(vals))` to coefficients in the space `s`.
Defaults to `coefficients(transform(canonicalspace(space),values),canonicalspace(space),space)`

# Examples
```jldoctest
julia> F = Fun(x -> x^2, Chebyshev());

julia> coefficients(F)
3-element Vector{Float64}:
 0.5
 0.0
 0.5

julia> transform(Chebyshev(), values(F)) ≈ coefficients(F)
true

julia> v = map(F, points(Chebyshev(), 4)); # custom grid

julia> transform(Chebyshev(), v)
4-element Vector{Float64}:
 0.5
 0.0
 0.5
 0.0
```
"""
transform(S::Space, vals) = plan_transform(S,vals)*vals

"""
    itransform(s::Space,coefficients::AbstractVector)

Transform coefficients back to values.  Defaults to using `canonicalspace` as in `transform`.

# Examples
```jldoctest
julia> F = Fun(x->x^2, Chebyshev())
Fun(Chebyshev(), [0.5, 0.0, 0.5])

julia> itransform(Chebyshev(), coefficients(F)) ≈ values(F)
true

julia> itransform(Chebyshev(), [0.5, 0, 0.5])
3-element Vector{Float64}:
 0.75
 0.0
 0.75
```
"""
itransform(S::Space, cfs) = plan_itransform(S,cfs)*cfs

itransform!(S::Space,cfs) = plan_itransform!(S,cfs)*cfs
transform!(S::Space,cfs) = plan_transform!(S,cfs)*cfs


_coefficients!!(::Val{true}) = coefficients!
_coefficients!!(::Val{false}) = coefficients
_multransform(P::CanonicalTransformPlan, ip, vals) = _coefficients!!(ip)(P.plan * vals, P.canonicalspace, P.space)
_multransform(P::ICanonicalTransformPlan, ip, cfs) = P.plan * _coefficients!!(ip)(cfs, P.space, P.canonicalspace)
*(P::Union{CanonicalTransformPlan, ICanonicalTransformPlan}, vals::AbstractVector) = _multransform(P, Val(inplace(P)), vals)


for OP in (:plan_transform,:plan_itransform,:plan_transform!,:plan_itransform!)
    # plan transform expects a vector
    # this passes an empty Float64 array
    @eval begin
        $OP(S::Space,::Type{T},n::Integer) where {T} = $OP(S,Vector{T}(undef, n))
        $OP(S::Space,n::Integer) = $OP(S, Float64, n)
    end
end

## sorting
# we sort spaces lexigraphically by default

for OP in (:<,:(<=),:(isless))
    @eval $OP(a::Space,b::Space)=$OP(string(a),string(b))
end

## Important special spaces


struct ZeroSpace{DD,R} <: Space{DD,R}
    domain::DD
    ZeroSpace{DD,R}(d::DD) where {DD,R} = new(d)
    ZeroSpace{DD,R}(d::AnyDomain) where {DD,R} = new(strictconvert(DD,d))
end


ZeroSpace(S::Space) = ZeroSpace{domaintype(S),rangetype(S)}(domain(S))
ZeroSpace() = ZeroSpace{AnyDomain,UnsetNumber}(AnyDomain())
domain(S::ZeroSpace) = S.domain

dimension(::ZeroSpace) = 0

spacescompatible(::ZeroSpace,::ZeroSpace) = true

pick_maybe_nonambiguous_space(a::UnsetSpace, b::ZeroSpace) = a
pick_maybe_nonambiguous_space(a::ZeroSpace, b::UnsetSpace) = b


"""
    ConstantSpace

The 1-dimensional scalar space.
"""
struct ConstantSpace{DD,R} <: Space{DD,R}
    domain::DD
    ConstantSpace{DD,R}(d::DD) where {DD,R} = new(d)
    ConstantSpace{DD,R}(d::AnyDomain) where {DD,R} = new(strictconvert(DD,d))
end

ConstantSpace(d::Domain) = ConstantSpace{typeof(d),real(prectype(d))}(d)

ConstantSpace(::Type{N},d::Domain) where {N<:Number} = ConstantSpace{typeof(d),real(N)}(d)
ConstantSpace(::Type{N}) where {N<:Number} = ConstantSpace(N,AnyDomain())
ConstantSpace() = ConstantSpace(Float64)


convert(::Type{Space}, z::Number) = ConstantSpace(strictconvert(Domain, z))  # Spaces
convert(::Type{ConstantSpace}, d::Domain) = ConstantSpace(d)
Space(z::Number) = strictconvert(Space, z)

isconstspace(::ConstantSpace) = true

for pl in (:plan_transform,:plan_transform!,:plan_itransform,:plan_itransform!)
    @eval $pl(sp::ConstantSpace,vals::AbstractVector) = I
end

# we override maxspace instead of maxspace_rule to avoid
# domainscompatible check.
for OP in (:maxspace,:(union))
    @eval begin
        $OP(A::ConstantSpace{AnyDomain},B::ConstantSpace{AnyDomain}) = A
        $OP(A::ConstantSpace{AnyDomain},B::ConstantSpace) = B
        $OP(A::ConstantSpace,B::ConstantSpace{AnyDomain}) = A
        $OP(A::ConstantSpace,B::ConstantSpace) = ConstantSpace(domain(A) ∪ domain(B))
    end
end

space(x::Number) = ConstantSpace(typeof(x))
_maybestaticsize(A) = size(A)
_maybestaticsize(A::SArray) = Val(size(A))
space(f::AbstractArray{T}) where T<:Number = ArraySpace(ConstantSpace(T), _maybestaticsize(f))

setdomain(A::ConstantSpace{DD,R}, d) where {DD,R} = ConstantSpace{typeof(d),R}(d)

blocklengths(::ConstantSpace) = SVector(1)

# Range type is Nothing since function evaluation is not defined
struct SequenceSpace <: Space{PositiveIntegers,Nothing} end

"""
    SequenceSpace

The space of all sequences, i.e., infinite vectors.
Also denoted ℓ⁰.
"""
SequenceSpace()


const ℓ⁰ = SequenceSpace()
dimension(::SequenceSpace) = ∞
domain(::SequenceSpace) = ℕ
spacescompatible(::SequenceSpace,::SequenceSpace) = true


## Boundary

boundary(S::Space) = boundary(domain(S))
