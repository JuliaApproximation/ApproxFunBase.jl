export Conversion

abstract type Conversion{T}<:Operator{T} end

struct ConcreteConversion{D<:Space,R<:Space,T} <: Conversion{T}
    domainspace::D
    rangespace::R
end


function ConcreteConversion(::Type{T}, a::Space,b::Space) where {T}
    ConcreteConversion{typeof(a),typeof(b),T}(a,b)
end
function ConcreteConversion(a::Space,b::Space)
    T = promote_type(rangetype(a),rangetype(b))
    ConcreteConversion(T, a, b)
end

function ConcreteConversion{D,R,T}(C::ConcreteConversion) where {D<:Space,R<:Space,T}
    ConcreteConversion{D,R,T}(strictconvert(D,C.domainspace), strictconvert(R, C.rangespace))
end

function Operator{T}(C::ConcreteConversion) where {T}
    ConcreteConversion(T, C.domainspace,C.rangespace)::Operator{T}
end

domainspace(C::ConcreteConversion)=C.domainspace
rangespace(C::ConcreteConversion)=C.rangespace



function _implementconversionerror(a, b)
    error("Implement Conversion from ", typeof(a), " to ", typeof(b))
end
function _defaultConversion(a, spa, b)
    if typeof(spa) == typeof(a)
        spb = canonicalspace(b)
        if typeof(spb) == typeof(a)
            _implementconversionerror(spb, b)
        elseif typeof(spb) == typeof(b)
            _implementconversionerror(a, spb)
        else
            return _defaultConversion(a, spb, b)
        end
    elseif typeof(spa) == typeof(b)
        _implementconversionerror(a, spa)
    end
    Conversion(a, spa, b)
end
function defaultConversion(a::Space,b::Space)
    if a==b
        Conversion(a)
    elseif conversion_type(a,b)==NoSpace()
        spa = canonicalspace(a)
        _defaultConversion(a, spa, b)
    else
        _implementconversionerror(a, b)
    end
end

"""
    hasconcreteconversion_canonical(sp::Space, ::Val{:forward})

Return `Conversion(sp, canonicalspace(sp))` is known statically to be a `ConcreteConversion`.
Assumed to be false by default.

    hasconcreteconversion_canonical(sp::Space, ::Val{:backward})

Return `Conversion(canonicalspace(sp), sp)` is known statically to be a `ConcreteConversion`.
Assumed to be false by default.
"""
hasconcreteconversion_canonical(@nospecialize(sp), @nospecialize(Valfwdback)) = false

function Conversion_maybeconcrete(sp, csp, v::Val{:forward})
    if hasconcreteconversion_canonical(sp, v)
        ConcreteConversion(sp,csp)
    else
        Conversion(sp,csp)
    end
end
function Conversion_maybeconcrete(sp, csp, v::Val{:backward})
    if hasconcreteconversion_canonical(sp, v)
        ConcreteConversion(csp,sp)
    else
        Conversion(csp,sp)
    end
end

"""
    Conversion(fromspace::Space, tospace::Space)

Represent a conversion operator between `fromspace` and `tospace`, when available.

See also [`PartialInverseOperator`](@ref) that might be able to represent the inverse,
even if this isn't banded.
"""
Conversion(a::Space,b::Space) = defaultConversion(a,b)
Conversion(a::Space) = ConversionWrapper(Operator(I,a))
Conversion() = ConversionWrapper(Operator(I,UnsetSpace()))


## Wrapper
# this allows for a Derivative implementation to return another operator, use a SpaceOperator containing
# the domain and range space
# but continue to know its a derivative

struct ConversionWrapper{D<:Space,R<:Space,T,O<:Operator{T}} <: Conversion{T}
    domainspace::D
    rangespace::R
    op::O
end

@wrapper ConversionWrapper false false

domainspace(C::ConversionWrapper) = C.domainspace
rangespace(C::ConversionWrapper) = C.rangespace

function ConversionWrapper(B::Operator,
        d::Space=domainspace(B), r::Space=rangespace(B))
    ConversionWrapper(d, r, unwrap(B))
end
Conversion(A::Space,B::Space,C::Space) =
    ConversionWrapper(Conversion(B,C)*Conversion(A,B), A, C)
Conversion(A::Space,B::Space,C::Space,D::Space...) =
    ConversionWrapper(Conversion(C,D...)*Conversion(B,C)*Conversion(A,B), A, last(D))

==(A::ConversionWrapper,B::ConversionWrapper) = A.op==B.op

function ConversionWrapper{D,R,T,O}(C::ConversionWrapper) where {D<:Space,R<:Space,T,O<:Operator{T}}
    ConversionWrapper{D,R,T,O}(
        strictconvert(D,C.domainspace),
        strictconvert(R,C.rangespace),
        strictconvert(O, C.op)
    )
end
function Operator{T}(D::ConversionWrapper) where T
    BO=strictconvert(Operator{T},D.op)
    d, r = domainspace(D), rangespace(D)
    ConversionWrapper(d, r, BO)::Operator{T}
end

convert(::Type{T}, C::ConversionWrapper) where {T<:Number} = strictconvert(T, C.op)


#promotedomainspace(P::Conversion,sp::Space)=ConversionWrapper(Operator(I, sp))
