export Multiplication

abstract type Multiplication{D,S,T} <:Operator{T} end

struct ConcreteMultiplication{D<:Space,S<:Space,T} <: Multiplication{D,S,T}
    f::VFun{D,T}
    space::S

    ConcreteMultiplication{D,S,T}(f::Fun{D,T},sp::S) where {D,S,T} = new{D,S,T}(f,sp)
end

function ConcreteMultiplication(::Type{V},f::Fun{D,T},sp::Space) where {V,D,T}
    if !domainscompatible(space(f),sp)
        error("Domain mismatch: cannot multiply function on $(domain(f)) to function on $(domain(sp))")
    end
    ConcreteMultiplication{D,typeof(sp),V}(
        convert(Fun{D,V},chop(f,40*eps(cfstype(f)))),sp)
end


function ConcreteMultiplication(f::Fun{D,T},sp::Space) where {D,T}
    if !domainscompatible(space(f),sp)
        error("Domain mismatch: cannot multiply function on $(domain(f)) to function on $(domain(sp))")
    end
    V = promote_type(T,rangetype(sp))
    ConcreteMultiplication{D,typeof(sp),V}(convert(Fun{D,V},chop(f,40*eps(cfstype(f)))),sp)
end

==(A::ConcreteMultiplication, B::ConcreteMultiplication) = (A.f == B.f) && (A.space == B.space)

# We do this in two stages to support Modifier spaces
# without ambiguity errors
function defaultMultiplication(f::Fun,sp::Space)
    csp=space(f)
    if csp==sp || !hasconversion(sp,csp)
        error("Implement Multiplication(::Fun{$(typeof(space(f)))},::$(typeof(sp)))")
    end
    MultiplicationWrapper(f,Multiplication(f,csp)*Conversion(sp,csp))
end

Multiplication(f::Fun,sp::Space) = defaultMultiplication(f,sp)


Multiplication(f::Fun,sp::UnsetSpace) = ConcreteMultiplication(f,sp)
Multiplication(f::Fun) = Multiplication(f,UnsetSpace())

Multiplication(c::Number,sp::Space) = Multiplication(Fun(c),sp)
Multiplication(sp::Space,c::Number) = Multiplication(sp,Fun(c))
Multiplication(c::Number) = Multiplication(Fun(c) )

# This covers right multiplication unless otherwise specified.
Multiplication(S::Space,f::Fun) = Multiplication(f,S)


function convert(::Type{Operator{T}},C::ConcreteMultiplication{S,V}) where {S,V,T}
    if T==eltype(C)
        C
    else
        ConcreteMultiplication{S,V,T}(Fun{S,T}(C.f),C.space)
    end
end

domainspace(M::ConcreteMultiplication{D,S,T}) where {D,S,T} = M.space
domain(T::ConcreteMultiplication) = domain(T.f)


## Default implementation: try converting to space of M.f

# avoid ambiguity
rangespace(D::ConcreteMultiplication{F,UnsetSpace,T}) where {F,T} = UnsetSpace()
getindex(D::ConcreteMultiplication{F,UnsetSpace,T},k::Integer,j::Integer) where {F,T} =
    error("No range space attached to Multiplication")






##multiplication can always be promoted, range space is allowed to change
promotedomainspace(D::Multiplication,sp::UnsetSpace) = D
promotedomainspace(D::Multiplication,sp::Space) = Multiplication(D.f,sp)
promoterangespace(D::ConcreteMultiplication{P,UnsetSpace},sp::UnsetSpace) where {P} = D
promoterangespace(D::ConcreteMultiplication{P,UnsetSpace},sp::Space) where {P} =
    promoterangespace(Multiplication(D.f,ConstantSpace(domain(sp))), sp)

choosedomainspace(M::ConcreteMultiplication{D,UnsetSpace},::UnsetSpace) where {D} = space(M.f)
# we assume multiplication maps spaces to themselves
choosedomainspace(M::ConcreteMultiplication{D,UnsetSpace},sp::Space) where {D} = sp


diagm(a::Fun) = Multiplication(a)

struct MultiplicationWrapper{D<:Space,S<:Space,O<:Operator,T} <: Multiplication{D,S,T}
    f::VFun{D,T}
    op::O
end

MultiplicationWrapper(T::Type,f::Fun{D,V},op::Operator) where {D<:Space,V} = MultiplicationWrapper{D,typeof(domainspace(op)),typeof(op),T}(f,op)
MultiplicationWrapper(f::Fun{D,V},op::Operator) where {D<:Space,V} = MultiplicationWrapper(eltype(op),f,op)

@wrapper MultiplicationWrapper

function convert(::Type{Operator{TT}},C::MultiplicationWrapper{S,V,O,T}) where {TT,S,V,O,T}
    if TT==T
        C
    else
        MultiplicationWrapper(Fun{S,TT}(C.f),Operator{TT}(C.op))::Operator{TT}
    end
end


## Multiplication operators allow us to multiply two spaces



hasfasttransform(_) = false
hasfasttransform(f::Fun)::Bool = hasfasttransform(space(f))
hasfasttransformtimes(f,g)::Bool = pointscompatible(f,g) && hasfasttransform(f) && hasfasttransform(g)


function default_mult_compatible(f::Fun, g::Fun)
    m,n = ncoefficients(f), ncoefficients(g)
    # Heuristic division of parameter space between value-space and coefficient-space multiplication.
    if log10(m)*log10(n)>4 && hasfasttransformtimes(f,g)
        transformtimes(f,g)
    elseif m≤n
        coefficienttimes(f,g)
    else
        coefficienttimes(g,f)
    end
end
# This should be overriden whenever the multiplication space is different
function default_mult(f::Fun,g::Fun)
    # When the spaces differ we promote and multiply
    if domainscompatible(space(f),space(g))
        default_mult_compatible(f, g)
    else
        sp=union(space(f),space(g))::Space
        fnew = Fun(f,sp)
        gnew = Fun(g,sp)
        default_mult_compatible(fnew, gnew)
    end
end

*(f::Fun,g::Fun) = default_mult(f,g)

coefficienttimes(f::Fun,g::Fun) = Multiplication(f,space(g))*g

function transformtimes(f::Fun,g::Fun, n = ncoefficients(f) + ncoefficients(g) - 1, sp = space(f))
    @assert pointscompatible(sp,space(g))::Bool
    iszero(ncoefficients(f)) && return f
    iszero(ncoefficients(g)) && return g
    f2,g2 = pad(f,n), pad(g,n)
    v = values(f2)
    v .*= values(g2)
    hc = transform(sp, v)
    chop!(Fun(sp,hc),10eps(eltype(hc)))
end


*(a::Fun,L::UniformScaling) = Multiplication(a*L.λ,UnsetSpace())
*(L::UniformScaling,a::Fun) = L.λ*a


## docs

"""
`Multiplication(f::Fun,sp::Space)` is the operator representing multiplication by
`f` on functions in the space `sp`.
"""
Multiplication(::Fun,::Space)

"""
`Multiplication(f::Fun)` is the operator representing multiplication by
`f` on an unset space of functions.  Spaces will be inferred when applying or
manipulating the operator.
"""
Multiplication(::Fun)
