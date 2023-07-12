export Ultraspherical, NormalizedUltraspherical

#Ultraspherical Spaces


"""
`Ultraspherical(λ)` is the space spanned by the ultraspherical polynomials
```
    C_0^{(λ)}(x),C_1^{(λ)}(x),C_2^{(λ)}(x),…
```
Note that `λ=1` this reduces to Chebyshev polynomials of the second kind:
`C_k^{(1)}(x) = U_k(x)`.
For `λ=1/2` this also reduces to Legendre polynomials:
`C_k^{(1/2)}(x) = P_k(x)`.
"""
struct Ultraspherical{T,D<:Domain,R} <: PolynomialSpace{D,R}
    order::T
    domain::D
    Ultraspherical{T,D,R}(m::T,d::D) where {T,D,R} = (@assert m ≠ 0; new(m,d))
    Ultraspherical{T,D,R}(m::Number,d::Domain) where {T,D,R} = (@assert m ≠ 0; new(strictconvert(T,m),strictconvert(D,d)))
    Ultraspherical{T,D,R}(d::Domain) where {T,D,R} = new(one(T),strictconvert(D,d))
    Ultraspherical{T,D,R}(m::Number) where {T,D,R} = (@assert m ≠ 0; new(strictconvert(T,m),D()))
end

Ultraspherical(m::Number,d::Domain) = Ultraspherical{typeof(m),typeof(d),real(prectype(d))}(m,d)
Ultraspherical(m::Number,d) = Ultraspherical(m,Domain(d))
Ultraspherical(m::Number) = Ultraspherical(m,ChebyshevInterval())
const NormalizedUltraspherical{T,D<:Domain,R} = NormalizedPolynomialSpace{Ultraspherical{T,D,R},D,R}
NormalizedUltraspherical(m) = NormalizedPolynomialSpace(Ultraspherical(m))
NormalizedUltraspherical(m,d) = NormalizedPolynomialSpace(Ultraspherical(m,d))


order(S::Ultraspherical) = S.order
order(N::NormalizedPolynomialSpace{<:Ultraspherical}) = order(N.space)
setdomain(S::Ultraspherical,d::Domain) = Ultraspherical(order(S),d)


convert(::Type{Ultraspherical{T,D,R}}, S::Ultraspherical{T,D,R}) where {T,D,R} = S
convert(::Type{Ultraspherical{A,D,R}}, S::Ultraspherical{B,D,R}) where {A,B,D,R} =
    Ultraspherical{A,D,R}(convert(A, order(S)), domain(S))

promote_rule(::Type{Ultraspherical{A,D,R}}, ::Type{Ultraspherical{B,D,R}}) where {A,B,D,R} =
    Ultraspherical{promote_type(A,B),D,R}


canonicalspace(S::Ultraspherical) = Chebyshev(domain(S))
pointscompatible(A::Ultraspherical, B::Chebyshev) = domain(A) == domain(B)
pointscompatible(A::Chebyshev, B::Ultraspherical) = domain(A) == domain(B)

struct UltrasphericalPlan{CT,FT,IP}
    chebplan::CT
    cheb2ultraplan::FT

    UltrasphericalPlan{CT,FT}(cp,c2lp,::Val{IP}) where {CT,FT,IP} = new{CT,FT,IP}(cp,c2lp)
end

struct UltrasphericalIPlan{CT,FT,IP}
    chebiplan::CT
    ultra2chebplan::FT

    UltrasphericalIPlan{CT,FT}(cp,c2lp,::Val{IP}) where {CT,FT,IP} = new{CT,FT,IP}(cp,c2lp)
end

function UltrasphericalPlan(λ::Number,vals,inplace = Val(false))
    cp = ApproxFunBase._plan_transform!!(inplace)(Chebyshev(),vals)
    c2lp = plan_cheb2ultra(vals, λ)
    UltrasphericalPlan{typeof(cp),typeof(c2lp)}(cp,c2lp,inplace)
end

function UltrasphericalIPlan(λ::Number,cfs,inplace = Val(false))
    cp = ApproxFunBase._plan_itransform!!(inplace)(Chebyshev(),cfs)
    l2cp = plan_ultra2cheb(cfs, λ)
    UltrasphericalIPlan{typeof(cp),typeof(l2cp)}(cp,l2cp,inplace)
end

*(UP::UltrasphericalPlan{<:Any,<:Any,false},v::AbstractVector) =
    UP.cheb2ultraplan*(UP.chebplan*v)
*(UP::UltrasphericalIPlan{<:Any,<:Any,false},v::AbstractVector) =
    UP.chebiplan*(UP.ultra2chebplan*v)

*(UP::UltrasphericalPlan{<:Any,<:Any,true},v::AbstractVector) =
    lmul!(UP.cheb2ultraplan, UP.chebplan*v)
*(UP::UltrasphericalIPlan{<:Any,<:Any,true},v::AbstractVector) =
    UP.chebiplan * lmul!(UP.ultra2chebplan, v)

plan_transform(sp::Ultraspherical{Int},vals::AbstractVector) = CanonicalTransformPlan(sp,vals)
plan_transform!(sp::Ultraspherical{Int},vals::AbstractVector) = CanonicalTransformPlan(sp,vals,Val(true))
plan_transform(sp::Ultraspherical,vals::AbstractVector) = UltrasphericalPlan(order(sp),vals)
plan_transform!(sp::Ultraspherical,vals::AbstractVector) = UltrasphericalPlan(order(sp),vals,Val(true))
plan_itransform(sp::Ultraspherical{Int},cfs::AbstractVector) = ICanonicalTransformPlan(sp,cfs)
plan_itransform!(sp::Ultraspherical{Int},cfs::AbstractVector) = ICanonicalTransformPlan(sp,cfs,Val(true))
plan_itransform(sp::Ultraspherical,cfs::AbstractVector) = UltrasphericalIPlan(order(sp),cfs)
plan_itransform!(sp::Ultraspherical,cfs::AbstractVector) = UltrasphericalIPlan(order(sp),cfs,Val(true))

## Construction

#domain(S) may be any domain

ones(::Type{T},S::Ultraspherical) where {T<:Number} = Fun(S,fill(one(T),1))
ones(S::Ultraspherical) = ones(Float64, S)



## Fast evaluation

function first(f::Fun{<:Ultraspherical{Int}})
    n = length(f.coefficients)
    n == 0 && return zero(cfstype(f))
    n == 1 && return first(f.coefficients)
    foldr(-,coefficients(f,Chebyshev))
end

last(f::Fun{<:Ultraspherical{Int}}) = reduce(+,coefficients(f,Chebyshev))

first(f::Fun{<:Ultraspherical}) = f(leftendpoint(domain(f)))
last(f::Fun{<:Ultraspherical}) = f(rightendpoint(domain(f)))

function Fun(::typeof(identity), s::Ultraspherical)
    d = domain(s)
    m = order(s)
    Fun(s, [mean(d), complexlength(d)/4m])
end


## Calculus




spacescompatible(a::Ultraspherical,b::Ultraspherical) =
    compare_orders(order(a), order(b)) && domainscompatible(a,b)
hasfasttransform(::Ultraspherical) = true

# these methods help with type-inference
hasconversion(C::Chebyshev, U::Ultraspherical{<:Union{Int,StaticInt}}) = domainscompatible(C,U)
hasconversion(U::Ultraspherical{<:Union{Int,StaticInt}}, C::Chebyshev) = false

include("UltrasphericalOperators.jl")
include("ultrasphericalconversions.jl")
include("fastops.jl")
