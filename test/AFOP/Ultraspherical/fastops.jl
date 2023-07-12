###
# This file contains BLAS/BandedMatrix overrides for operators
# that depend on the structure of BandedMatrix
####




#####
# Conversions
#####

function BandedMatrix(S::SubOperator{T,ConcreteConversion{Chebyshev{DD,RR},
                Ultraspherical{LT,DD,RR},T},
            NTuple{2,UnitRange{Int}}}) where {T,LT<:Union{Integer, StaticInt},DD,RR}
    # we can assume order is 1
    ret = BandedMatrix{eltype(S)}(undef, size(S), bandwidths(S))
    kr,jr = parentindices(S)
    dg = diagindshift(S)

    @assert -bandwidth(ret,1) ≤ dg ≤ bandwidth(ret,2)-2

    ret[band(dg)] .= 0.5
    ret[band(dg+1)] .= 0.0
    ret[band(dg+2)] .= -0.5

    # correct first entry
    if 1 in kr && 1 in jr
        ret[1,1] = 1.0
    end

    ret
end

function BandedMatrix(V::SubOperator{T,ConcreteConversion{Ultraspherical{LT1,DD,RR},Ultraspherical{LT2,DD,RR},T},
                                                                  NTuple{2,UnitRange{Int}}}) where {T,LT1,LT2,DD,RR}

    n,m = size(V)
    V_l, V_u = bandwidths(V)
    ret = BandedMatrix{eltype(V)}(undef, (n,m), (V_l,V_u))
    kr,jr = parentindices(V)

    (isempty(kr) || isempty(jr)) && return ret

    dg = diagindshift(V)


    λ = order(rangespace(parent(V)))
    c = λ-one(T)

    # need to drop columns



    1-n ≤ dg ≤ m-1 && (ret[band(dg)] .= c./(jr[max(0,dg)+1:min(n+dg,m)] .- 2 .+ λ))
    1-n ≤ dg+1 ≤ m-1 && (ret[band(dg+1)] .= 0)
    1-n ≤ dg+2 ≤ m-1 && (ret[band(dg+2)] .= c./(2 .- λ .- jr[max(0,dg+2)+1:min(n+dg+2,m)]))

    ret
end


#####
# Derivatives
#####



function BandedMatrix(S::SubOperator{T,ConcreteDerivative{Chebyshev{DD,RR},K,T},
                                                     NTuple{2,UnitRange{Int}}}) where {T,K,DD,RR}

    n,m = size(S)
    ret = BandedMatrix{eltype(S)}(undef, (n,m), bandwidths(S))
    kr,jr = parentindices(S)
    dg = diagindshift(S)

    D = parent(S)
    k = D.order
    d = domain(D)

    C=strictconvert(T,pochhammer(one(T),k-1)/2*(4/(complexlength(d)))^k)


    # need to drop columns


    if 1-n ≤ dg+k ≤ m-1
        ret[band(dg+k)] .= C.*(jr[max(0,dg+k)+1:min(n+dg+k,m)] .- one(T))
    end

    ret
end


function BandedMatrix(S::SubOperator{T,ConcreteDerivative{Ultraspherical{LT,DD,RR},K,T},
                                                  NTuple{2,UnitRange{Int}}}) where {T,K,DD,RR,LT}
    n,m = size(S)
    ret = BandedMatrix{eltype(S)}(undef, (n,m), bandwidths(S))
    kr,jr = parentindices(S)
    dg = diagindshift(S)

    D = parent(S)
    k = D.order
    λ = order(domainspace(D))
    d = domain(D)

    C = strictconvert(T,pochhammer(one(T)*λ,k)*(4/(complexlength(d)))^k)
    ret[band(dg+k)] .= C

    ret
end

