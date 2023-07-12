##  Jacobi Operator




recA(::Type{T},S::Ultraspherical,k) where {T} = (2*(k+order(S)))/(k+one(T))   # one(T) ensures we get correct type
recB(::Type{T},::Ultraspherical,_) where {T} = zero(T)
recC(::Type{T},S::Ultraspherical,k) where {T} = (k-one(T)+2order(S))/(k+one(T))   # one(T) ensures we get correct type

# x p_k
recα(::Type{T},::Ultraspherical,_) where {T} = zero(T)
recβ(::Type{T},S::Ultraspherical,k) where {T} = k/(2*(k-one(T)+order(S)))   # one(T) ensures we get correct type
recγ(::Type{T},S::Ultraspherical,k) where {T} = (k-2+2order(S))/(2*(k-one(T)+order(S)))   # one(T) ensures we get correct type


function normalization(::Type{T}, sp::Ultraspherical, k::Int) where T
    λ = order(sp)
    T(2)^(1-2λ)*π/((k+λ)*gamma(λ)^2*FastTransforms.Λ(real(T(k)),one(λ),2λ))
end

## Multiplication
# these are special cases


Base.stride(M::ConcreteMultiplication{<:Chebyshev,<:Ultraspherical}) =
    stride(M.f)
Base.stride(M::ConcreteMultiplication{<:Ultraspherical,<:Chebyshev}) =
    stride(M.f)
Base.stride(M::ConcreteMultiplication{<:Ultraspherical,<:Ultraspherical}) =
    stride(M.f)

@inline function _Multiplication(f::Fun{<:Chebyshev}, sp::Ultraspherical{<:Union{Integer,StaticInt}})
    if order(sp) == 1
        cfs = f.coefficients
        MultiplicationWrapper(f,
            SpaceOperator(
                SymToeplitzOperator(cfs/2) +
                    HankelOperator(view(cfs,3:length(cfs))/(-2)),
                sp, sp)
        )
    else
        ConcreteMultiplication(f,sp)
    end
end
@static if VERSION >= v"1.8"
    Base.@constprop aggressive Multiplication(f::Fun{<:Chebyshev}, sp::Ultraspherical{<:Union{Integer,StaticInt}}) =
        _Multiplication(f, sp)
else
    Multiplication(f::Fun{<:Chebyshev}, sp::Ultraspherical{<:Union{Integer,StaticInt}}) = _Multiplication(f, sp)
end


## Derivative


#Derivative(k::Integer,d::IntervalOrSegment)=Derivative(k-1:k,d)
#Derivative(d::IntervalOrSegment)=Derivative(1,d)


function Derivative(sp::Ultraspherical{LT,DD}, m::Number) where {LT,DD<:IntervalOrSegment}
    assert_integer(m)
    ConcreteDerivative(sp,m)
end
function Integral(sp::Ultraspherical{<:Any,<:IntervalOrSegment}, m::Number)
    assert_integer(m)
    λ = order(sp)
    if m ≤ λ
        ConcreteIntegral(sp,m)
    else # Convert up
        nsp = Ultraspherical(m,domain(sp))
        Integralop = ConcreteIntegral(nsp,m)
        C = Conversion(sp,nsp)
        IntegralWrapper(Integralop * C, m, sp, rangespace(Integralop))
    end
end


rangespace(D::ConcreteDerivative{<:Ultraspherical{LT,DD}}) where {LT,DD<:IntervalOrSegment} =
    Ultraspherical(order(domainspace(D))+D.order,domain(D))

bandwidths(D::ConcreteDerivative{<:Ultraspherical{LT,DD}}) where {LT,DD<:IntervalOrSegment} = -D.order,D.order
bandwidths(D::ConcreteIntegral{<:Ultraspherical{LT,DD}}) where {LT,DD<:IntervalOrSegment} = D.order,-D.order
Base.stride(D::ConcreteDerivative{<:Ultraspherical{LT,DD}}) where {LT,DD<:IntervalOrSegment} = D.order

isdiag(D::ConcreteDerivative{<:Ultraspherical{<:Any,<:IntervalOrSegment}}) = false
isdiag(D::ConcreteIntegral{<:Ultraspherical{<:Any,<:IntervalOrSegment}}) = false

function getindex(D::ConcreteDerivative{<:Ultraspherical{TT,DD},K,T},
               k::Integer,j::Integer) where {TT,DD<:IntervalOrSegment,K,T}
    m=D.order
    d=domain(D)
    λ=order(domainspace(D))

    if j==k+m
        strictconvert(T,(pochhammer(one(T)*λ,m)*(4/complexlength(d)).^m))
    else
        zero(T)
    end
end


## Integral

linesum(f::Fun{<:Ultraspherical{LT,DD}}) where {LT,DD<:IntervalOrSegment} =
    sum(setcanonicaldomain(f))*arclength(d)/2


function rangespace(D::ConcreteIntegral{<:Ultraspherical{LT,DD}}) where {LT,DD<:IntervalOrSegment}
    k = order(domainspace(D))-D.order
    k == 0 ? Chebyshev(domain(D)) : Ultraspherical(k, domain(D))
end

function getindex(Q::ConcreteIntegral{<:Ultraspherical{LT,DD}},k::Integer,j::Integer) where {LT,DD<:IntervalOrSegment}
    T=eltype(Q)
    m=Q.order
    d=domain(Q)
    λ=order(domainspace(Q))
    @assert m<=λ

    if λ == 1 && k==j+1
        C = complexlength(d)/2
        strictconvert(T, C/(k-1))
    elseif λ == m && k == j + m
        C = complexlength(d)/2
        U1toC = C/(k-1)
        UmtoU1 = pochhammer(one(T)*λ,-(m-1))*(complexlength(d)/4)^(m-1)
        strictconvert(T, U1toC * UmtoU1)
    elseif λ > 1 && k==j+m
        strictconvert(T,pochhammer(one(T)*λ,-m)*(complexlength(d)/4)^m)
    else
        zero(T)
    end
end



## Conversion Operator

function Conversion(A::Chebyshev, B::Ultraspherical)
    @assert domain(A) == domain(B)
    mB = order(B)
    d=domain(A)
    dB = domain(B)
    if isequalhalf(mB) || mB == 1
        return ConcreteConversion(A,B)
    elseif (isinteger(mB) || isapproxhalfoddinteger(mB)) && mB > 0
        r = mB:-1:(isinteger(mB) ? 2 : 1)
        v = [ConcreteConversion(Ultraspherical(i-1, d), Ultraspherical(i,d)) for i in r]
        U = domainspace(last(v))
        CAU = ConcreteConversion(A, U)
        v2 = Union{eltype(v), typeof(CAU)}[v; CAU]
        bwsum = isapproxinteger(mB) ? (0, 2length(v2)) : (0,ℵ₀)
        return ConversionWrapper(TimesOperator(v2, bwsum, (ℵ₀,ℵ₀), bwsum), A, B)
    end
    throw(ArgumentError("please implement $A → $B"))
end

function Conversion(A::Ultraspherical,B::Chebyshev)
    @assert domain(A) == domain(B)
    if isequalhalf(order(A)) && domain(A) == domain(B)
        return ConcreteConversion(A,B)
    end
    throw(ArgumentError("please implement $A → $B"))
end


maxspace_rule(A::Ultraspherical,B::Chebyshev) = A


function Conversion(A::Ultraspherical,B::Ultraspherical)
    @assert domain(A) == domain(B)
    a=order(A); b=order(B)
    d=domain(A)
    if b==a
        return ConversionWrapper(Operator(I,A))
    elseif isapproxinteger(b-a) || isapproxhalfoddinteger(b-a)
        if -1 ≤ b-a ≤ 1 && (a,b) ≠ (2,1)
            return ConcreteConversion(A,B)
        elseif b-a > 1
            r = b:-1:a+1
            v = [ConcreteConversion(Ultraspherical(i-1,d), Ultraspherical(i,d)) for i in r]
            if !(last(r) ≈ a+1)
                vlast = ConcreteConversion(A, Ultraspherical(last(r)-1, d))
                v2 = Union{eltype(v), typeof(vlast)}[v; vlast]
            else
                v2 = v
            end
            bwsum = isapproxinteger(b-a) ? (0, 2length(v)) : (0,ℵ₀)
            return ConversionWrapper(TimesOperator(v2, bwsum, (ℵ₀,ℵ₀), bwsum), A, B)
        end
    end
    throw(ArgumentError("please implement $A → $B"))
end

function maxspace_rule(A::Ultraspherical, B::Ultraspherical)
    domainscompatible(A, B) || return NoSpace()
    isapproxinteger(order(A) - order(B)) || return NoSpace()
    order(A) > order(B) ? A : B
end

function getindex(M::ConcreteConversion{<:Chebyshev,U,T},
        k::Integer,j::Integer) where {T, U<:Ultraspherical{<:Union{Integer, StaticInt}}}
   # order must be 1
    if k==j==1
        one(T)
    elseif k==j
        one(T)/2
    elseif j==k+2
        -one(T)/2
    else
        zero(T)
    end
end


function getindex(M::ConcreteConversion{U1,U2,T},
        k::Integer,j::Integer) where {DD,RR,
            U1<:Ultraspherical{<:Union{Integer, StaticInt},DD,RR},
            U2<:Ultraspherical{<:Union{Integer, StaticInt},DD,RR},T}
    #  we can assume that λ==m+1
    λ=order(rangespace(M))
    c=λ-one(T)  # this supports big types
    if k==j
        c/(k - 2 + λ)
    elseif j==k+2
        -c/(k + λ)
    else
        zero(T)
    end
end

function getindex(M::ConcreteConversion{U1,U2,T},
        k::Integer,j::Integer) where {DD,RR,
            U1<:Ultraspherical{<:Any,DD,RR},
            U2<:Ultraspherical{<:Any,DD,RR},T}
    λ=order(rangespace(M))
    if order(domainspace(M))+1==λ
        c=λ-one(T)  # this supports big types
        if k==j
            c/(k - 2 + λ)
        elseif j==k+2
            -c/(k + λ)
        else
            zero(T)
        end
    else
        error("Not implemented")
    end
end


bandwidths(C::ConcreteConversion{<:Chebyshev,<:Ultraspherical{<:Union{Integer,StaticInt}}}) = 0,2  # order == 1
bandwidths(C::ConcreteConversion{<:Ultraspherical{<:Union{Integer,StaticInt}},<:Ultraspherical{<:Union{Integer,StaticInt}}}) = 0,2

function bandwidths(C::ConcreteConversion{<:Chebyshev,<:Ultraspherical})
    orderone = order(rangespace(C)) == 1
    orderone ? (0,2) : (0,ℵ₀)
end
function bandwidths(C::ConcreteConversion{<:Ultraspherical,<:Chebyshev})
    orderone = order(domainspace(C)) == 1
    orderone ? (0,2) : (0,ℵ₀)
end

function bandwidths(C::ConcreteConversion{<:Ultraspherical,<:Ultraspherical})
    offbyone = order(domainspace(C))+1 == order(rangespace(C))
    offbyone ? (0,2) : (0,ℵ₀)
end

Base.stride(C::ConcreteConversion{<:Chebyshev,<:Ultraspherical{<:Union{Integer,StaticInt}}}) = 2
Base.stride(C::ConcreteConversion{<:Ultraspherical,<:Ultraspherical}) = 2

isdiag(::ConcreteConversion{<:Chebyshev,<:Ultraspherical}) = false
isdiag(::ConcreteConversion{<:Ultraspherical,<:Chebyshev}) = false
isdiag(::ConcreteConversion{<:Ultraspherical,<:Ultraspherical}) = false

## coefficients

# return the space that has banded Conversion to the other
function conversion_rule(a::Chebyshev,b::Ultraspherical{<:Union{Integer,StaticInt}})
    if domainscompatible(a,b)
        a
    else
        NoSpace()
    end
end

conversion_rule(a::Ultraspherical{<:StaticInt}, b::Ultraspherical{<:StaticInt}) =
    _conversion_rule(a, b)
function conversion_rule(a::Ultraspherical{LT},b::Ultraspherical{LT}) where {LT}
    _conversion_rule(a, b)
end
function _conversion_rule(a::Ultraspherical, b::Ultraspherical)
    if domainscompatible(a,b) && isapproxinteger(order(a)-order(b))
        order(a) < order(b) ? a : b
    else
        NoSpace()
    end
end

# TODO: include in getindex to speed up
function Integral(sp::Chebyshev{DD}, m::Number) where {DD<:IntervalOrSegment}
    assert_integer(m)
    usp = Ultraspherical(m,domain(sp))
    I = Integral(usp,m)
    C = Conversion(sp,usp)
    T = TimesOperator(I, C)
    IntegralWrapper(T, m, sp, sp)
end



## Non-banded conversions

function getindex(M::ConcreteConversion{<:Chebyshev,<:Ultraspherical,T}, k::Integer,j::Integer) where {T}
    λ = order(rangespace(M))
    if λ == 1
        if k==j==1
            one(T)
        elseif k==j
            one(T)/2
        elseif j==k+2
            -one(T)/2
        else
            zero(T)
        end
    elseif λ == 0.5
        # Cheb-to-Leg
        if j==k==1
            one(T)
        elseif j==k
            strictconvert(T,sqrt(π)/(2FastTransforms.Λ(k-1)))
        elseif k < j && iseven(k-j)
            strictconvert(T,-(j-1)*(k-0.5)*(FastTransforms.Λ((j-k-2)/2)/(j-k))*
                            (FastTransforms.Λ((j+k-3)/2)/(j+k-1)))
        else
            zero(T)
        end
    else
        error("Not implemented")
    end
end


function getindex(M::ConcreteConversion{<:Ultraspherical,<:Chebyshev,T}, k::Integer,j::Integer) where {T}
    λ = order(domainspace(M))
    if λ == 1
        # order must be 1
        if k==j==1
            one(T)
        elseif k==j
            one(T)/2
        elseif j==k+2
            -one(T)/2
        else
            zero(T)
        end
    elseif λ == 0.5
        if k==1 && isodd(j)
            strictconvert(T,FastTransforms.Λ((j-1)/2)^2/π)
        elseif k ≤ j && iseven(k-j)
            strictconvert(T,FastTransforms.Λ((j-k)/2)*FastTransforms.Λ((k+j-2)/2)*2/π)
        else
            zero(T)
        end
    else
        error("Not implemented")
    end
end



function getindex(M::ConcreteConversion{Ultraspherical{LT,DD,RR},
                                     Ultraspherical{LT2,DD,RR},T},
                     k::Integer,j::Integer) where {DD,RR,LT,LT2,T}
    λ1 = order(domainspace(M))
    λ2 = order(rangespace(M))
    if abs(λ1-λ2) < 1
        if j ≥ k && iseven(k-j)
            strictconvert(T,(λ1 < λ2 && k ≠ j ? -1 : 1) *  # fix sign for lgamma
                exp(lgamma(λ2)+log(k-1+λ2)-lgamma(λ1)-lgamma(λ1-λ2) + lgamma((j-k)/2+λ1-λ2)-
                lgamma((j-k)/2+1)+lgamma((k+j-2)/2+λ1)-lgamma((k+j-2)/2+λ2+1)))
        else
            zero(T)
        end
    else
        error("Not implemented")
    end
end


ReverseOrientation(S::Ultraspherical) = ReverseOrientationWrapper(NegateEven(S,reverseorientation(S)))
Reverse(S::Ultraspherical) = ReverseWrapper(NegateEven(S,S))

Evaluation(S::MaybeNormalized{<:Ultraspherical},x::Number,o::Integer) = ConcreteEvaluation(S,x,o)
