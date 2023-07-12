export Jacobi, NormalizedJacobi, Legendre, NormalizedLegendre

"""
`Jacobi(b,a)` represents the space spanned by Jacobi polynomials `P_k^{(a,b)}`,
which are orthogonal with respect to the weight `(1+x)^β*(1-x)^α`
"""
struct Jacobi{D<:Domain,R,T} <: PolynomialSpace{D,R}
    b::T
    a::T
    domain::D
    Jacobi{D,R}(b::T,a::T,d::D) where {D,R,T<:Number} = new{D,R,T}(b,a,d)
end
Jacobi(b::T,a::T,d::Domain) where {T<:Number} =
    Jacobi{typeof(d),promote_type(T,real(prectype(d)))}(b, a, d)
Legendre(domain = ChebyshevInterval()) = Jacobi(0,0,Domain(domain)::Domain)
Legendre(s::PolynomialSpace) = Legendre(Jacobi(s))
Legendre(s::Jacobi) = s.a == s.b == 0 ? s : throw(ArgumentError("can't convert $s to Legendre"))
Jacobi(b::Number,a::Number,d=ChebyshevInterval()) = Jacobi(promote(b, a)..., Domain(d)::Domain)
function Jacobi(A::Ultraspherical)
    m = order(A)
    n = m + half(Odd(-1))
    Jacobi(n,n,domain(A))
end
function Jacobi(A::Chebyshev)
    n = half(Odd(-1))
    Jacobi(n,n,domain(A))
end
Jacobi(A::Jacobi) = A

const NormalizedJacobi{D<:Domain,R,T} = NormalizedPolynomialSpace{Jacobi{D,R,T},D,R}
NormalizedJacobi(s...) = NormalizedPolynomialSpace(Jacobi(s...))
NormalizedJacobi(s::Space) = NormalizedPolynomialSpace(Jacobi(_stripnorm(s)))
NormalizedLegendre(d...) = NormalizedPolynomialSpace(Legendre(d...))
NormalizedLegendre(s::Space) = NormalizedPolynomialSpace(Legendre(_stripnorm(s)))

function normalization(::Type{T}, sp::Jacobi, k::Int) where T
    x = FastTransforms.Anαβ(k, sp.a, sp.b)
    if sp.a == sp.b == -0.5 && k == 0
        # In this case, the expression for Anαβ has a division by zero, so we evaluate this using Mathematica
        # In principle this may be generalized to arbitrary α + β = -1
        # The exact expression from Mathematica in terms of the hypergeometric 2F1 function
        # \fbox{$\frac{\, _2F_1(1,-\alpha ;\beta +2;-1)}{\beta +1}+\frac{\, _2F_1(1,-\beta ;\alpha +2;-1)}{\alpha +1}\text{ if }\alpha >-1\land \beta >-1$}
        # or, by eliminating β and expressed in terms of the polygamma function,
        # \fbox{$\frac{1}{2} \left(\psi ^{(0)}\left(\frac{1}{2}-\frac{\alpha }{2}\right)-\psi ^{(0)}\left(-\frac{\alpha }{2}\right)\right)+\frac{1}{2} \left(\psi ^{(0)}\left(\frac{\alpha }{2}+1\right)-\psi ^{(0)}\left(\frac{\alpha }{2}+\frac{1}{2}\right)\right)\text{ if }-1<\alpha <0$}
        oftype(x, pi)
    else
        x
    end
end

function Ultraspherical(J::Jacobi)
    if J.a == J.b
        Ultraspherical(J.a+_onehalf(J.a),domain(J))
    else
        error("Cannot construct Ultraspherical with a=$(J.a) and b=$(J.b)")
    end
end
NormalizedUltraspherical(NS::NormalizedPolynomialSpace{<:Jacobi}) =
    NormalizedPolynomialSpace(Ultraspherical(NS.space))
NormalizedJacobi(NS::NormalizedPolynomialSpace{<:Union{Ultraspherical, Chebyshev}}) =
    NormalizedPolynomialSpace(Jacobi(NS.space))

Base.promote_rule(::Type{Jacobi{D,R1,T1}},::Type{Jacobi{D,R2,T2}}) where {D,R1,R2,T1,T2} =
    Jacobi{D,promote_type(R1,R2),promote_type(T1,T2)}
convert(::Type{Jacobi{D,R1,T1}},J::Jacobi{D,R2,T2}) where {D,R1,R2,T1,T2} =
    Jacobi{D,R1}(T1(J.b)::T1, T1(J.a)::T1, J.domain)::Jacobi{D,R1,T1}

compare_orders((Aa, Ba)::NTuple{2,Number}, (Ab, Bb)::NTuple{2,Number}) = compare_orders(Aa, Ba) && compare_orders(Ab, Bb)
spacescompatible(a::Jacobi, b::Jacobi) = compare_orders((a.a, b.a), (a.b, b.b)) && domainscompatible(a,b)

function canonicalspace(S::Jacobi)
    if isapproxhalfoddinteger(S.a) && isapproxhalfoddinteger(S.b)
        Chebyshev(domain(S))
    else
        # return space with parameters in (-1,0.]
        Jacobi(mod(S.b,-1),mod(S.a,-1),domain(S))
    end
end

reverseorientation(S::Jacobi) = Jacobi(S.a, S.b, reverseorientation(domain(S)))
ReverseOrientation(S::Jacobi) = ReverseOrientationWrapper(NegateEven(S,reverseorientation(S)))
reverseorientation(f::Fun{<:Jacobi}) =
    Fun(reverseorientation(space(f)), alternatesign!(copy(f.coefficients)))


#####
# jacobirecA/B/C is from dlmf:
# p_{n+1} = (A_n x + B_n)p_n - C_n p_{n-1}
#####
@inline function jacobirecA(::Type{T},α,β,k)::T where T
    if k==0 && ((α+β==0)||(α+β==-1))
        (α+β)/2+one(T)
    else
        (2k+α+β+one(T))*(2k+α+β+2one(T))/(2*(k+one(T))*(k+α+β+one(T)))
    end
end
@inline function jacobirecB(::Type{T},α,β,k)::T where T
    if k==0 && ((α+β==0)||(α+β==-1))
        (α-β)*one(T)/2
    else
        (α-β)*(α+β)*(2k+α+β+one(T))/(2*(k+one(T))*(k+α+β+one(T))*(2one(T)*k+α+β))
    end
end
@inline function jacobirecC(::Type{T},α,β,k)::T where T
    (one(T)*k+α)*(one(T)*k+β)*(2k+α+β+2one(T))/((k+one(T))*(k+α+β+one(T))*(2one(T)*k+α+β))
end
#####
# jacobirecA/B/C is from dlmf:
# x p_{n-1} =γ_n p_{n-2} + α_n p_{n-1} +  p_n β_n
#####

@inline jacobirecγ(::Type{T},α,β,k) where {T} = jacobirecC(T,α,β,k-1)/jacobirecA(T,α,β,k-1)
@inline jacobirecα(::Type{T},α,β,k) where {T} = -jacobirecB(T,α,β,k-1)/jacobirecA(T,α,β,k-1)
@inline jacobirecβ(::Type{T},α,β,k) where {T} = 1/jacobirecA(T,α,β,k-1)

for (REC,JREC) in ((:recα,:jacobirecα),(:recβ,:jacobirecβ),(:recγ,:jacobirecγ),
                   (:recA,:jacobirecA),(:recB,:jacobirecB),(:recC,:jacobirecC))
    @eval @inline $REC(::Type{T},sp::Jacobi,k) where {T} = $JREC(T,sp.a,sp.b,k)::T
end


function jacobip(::Type{T},r::AbstractRange,α,β,x::Number) where T
    if x==1 && α==0
        fill(one(T), length(r))
    elseif x==-1 && β==0
        (-one(T)).^r
    elseif isempty(r)
        T[]
    else
        n=r[end]+1
        if n<=2
            v=T[1,(α-β+(2+α+β)*x)/2]
        else
            v=Vector{T}(undef, n)  # x may be complex
            v[1]=1
            v[2]=(α-β+(2+α+β)*x)/2

            @inbounds for k=2:n-1
                v[k+1]=(jacobirecA(T,α,β,k-1)*x+jacobirecB(T,α,β,k-1))*v[k] - jacobirecC(T,α,β,k-1)*v[k-1]
            end
        end
        v[r.+1]
    end
end


jacobip(r::AbstractRange,α,β,x::Number) = jacobip(promote_type(typeof(α),typeof(β),typeof(x)),r,α,β,x)

jacobip(::Type{T},n::Integer,α,β,v) where {T} = jacobip(T,n:n,α,β,v)[1]
jacobip(n::Integer,α,β,v) = jacobip(n:n,α,β,v)[1]
jacobip(::Type{T},n,S::Jacobi,v) where {T} = jacobip(T,n,S.a,S.b,v)
jacobip(n,S::Jacobi,v) = jacobip(n,S.a,S.b,v)





include("jacobitransform.jl")
include("JacobiOperators.jl")




one(::Type{T},S::Jacobi) where {T<:Number} = Fun(S,fill(one(T),1))
one(S::Jacobi) = Fun(S,fill(one(Float64),1))
zeros(::Type{T},S::Jacobi) where {T<:Number} = Fun(S,zeros(T,1))
zeros(S::Jacobi) = Fun(S,zeros(prectype(S),1))

_Fun(J::Jacobi, ::ChebyshevInterval) = Fun(J, [(J.b-J.a)/(2+J.a+J.b),2.0/(2+J.a+J.b)])
_unscaledcoeff(J) = [2.0*(J.b + 1)/(2+J.a+J.b), 2.0/(2+J.a+J.b)]
function _Fun(J::Jacobi, d::Interval)
    scale = complexlength(d)/2
    coeffs = _unscaledcoeff(J) .* scale
    coeffs[1] += leftendpoint(d)
    Fun(J, coeffs)
end
function _Fun(J::Jacobi, d)
    complexlength(d)/2*(Fun(J,_unscaledcoeff(J)))+leftendpoint(d)
end

function Fun(::typeof(identity), J::Jacobi)
    _Fun(J, domain(J))
end


setdomain(S::Jacobi,d::Domain)=Jacobi(S.b,S.a,d)


# O(min(m,n)) Jacobi conjugated inner product

function conjugatedinnerproduct(sp::Jacobi,u::AbstractVector{S},v::AbstractVector{V}) where {S,V}
    T,mn = promote_type(S,V),min(length(u),length(v))
    α,β = sp.a,sp.b
    if mn > 1
        wi = 2^(α+β+1)*gamma(α+1)*gamma(β+1)/gamma(α+β+2)
        ret = u[1]*wi*v[1]
        for i=2:mn
            wi *= (α+i-1)*(β+i-1)/(i-1)/(i-1+α+β)*(2i+α+β-3)/(2i+α+β-1)
            ret += u[i]*wi*v[i]
        end
        return ret
    elseif mn > 0
        wi = 2^(α+β+1)*gamma(α+1)*gamma(β+1)/gamma(α+β+2)
        return u[1]*wi*v[1]
    else
        return zero(promote_type(eltype(u),eltype(v)))
    end
end

function bilinearform(f::Fun{J},g::Fun{J}) where {J<:Jacobi}
    @assert domain(f) == domain(g)
    if f.space.a == g.space.a == 0. && f.space.b == g.space.b == 0.
        return complexlength(domain(f))/2*conjugatedinnerproduct(g.space,f.coefficients,g.coefficients)
    else
        return defaultbilinearform(f,g)
    end
end


function linebilinearform(f::Fun{J},g::Fun{J}) where {J<:Jacobi}
    @assert domain(f) == domain(g)
    if f.space.a == g.space.a == 0. && f.space.b == g.space.b == 0.
        return arclength(domain(f))/2*conjugatedinnerproduct(g.space,f.coefficients,g.coefficients)
    else
        return defaultlinebilinearform(f,g)
    end
end
