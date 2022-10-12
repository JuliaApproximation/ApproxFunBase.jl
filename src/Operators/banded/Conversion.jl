export Conversion

abstract type Conversion{T}<:Operator{T} end

struct ConcreteConversion{S<:Space,V<:Space,T} <: Conversion{T}
    domainspace::S
    rangespace::V
end


ConcreteConversion(a::Space,b::Space)=ConcreteConversion{typeof(a),typeof(b),
        promote_type(rangetype(a),rangetype(b))}(a,b)


function convert(::Type{Operator{T}},C::ConcreteConversion{S,V}) where {T,S,V}
    if T==eltype(C)
        C
    else
        ConcreteConversion{S,V,T}(C.domainspace,C.rangespace)
    end
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

struct ConversionWrapper{S<:Operator,T} <: Conversion{T}
    op::S
end

@wrapper ConversionWrapper


ConversionWrapper(::Type{T},op) where {T} = ConversionWrapper{typeof(op),T}(op)
ConversionWrapper(B::Operator) =
    ConversionWrapper{typeof(B),eltype(B)}(B)
Conversion(A::Space,B::Space,C::Space) =
    ConversionWrapper(Conversion(B,C)*Conversion(A,B))
Conversion(A::Space,B::Space,C::Space,D::Space...) =
    ConversionWrapper(Conversion(C,D...)*Conversion(B,C)*Conversion(A,B))

==(A::ConversionWrapper,B::ConversionWrapper) = A.op==B.op


function convert(::Type{Operator{T}},D::ConversionWrapper) where T
    if T==eltype(D)
        D
    else
        BO=strictconvert(Operator{T},D.op)
        ConversionWrapper{typeof(BO),T}(BO)
    end
end



#promotedomainspace(P::Conversion,sp::Space)=ConversionWrapper(Operator(I, sp))
