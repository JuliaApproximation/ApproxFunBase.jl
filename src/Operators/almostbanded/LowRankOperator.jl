export AbstractLowRankOperator, LowRankOperator

abstract type AbstractLowRankOperator{T} <: Operator{T} end

struct LowRankOperator{S<:Space,T,O<:Operator{T}} <: AbstractLowRankOperator{T}
    U::Vector{VFun{S,T}}
    V::Vector{O}

    function LowRankOperator{S,T}(U::Vector{VFun{S,T}}, V::Vector{O}) where {S,T,O<:Operator{T}}
        @assert all(isafunctional,V)

        @assert length(U) == length(V)
        @assert length(U) > 0
        ds=domainspace(first(V))
        for k=2:length(V)
            @assert domainspace(V[k])==ds
        end
        rs=space(first(U))
        for k=2:length(U)
            @assert space(U[k])==rs
        end
        new{S,T,O}(U,V)
    end
end



LowRankOperator(U::Vector{VFun{S,T}}, V::Vector{<:Operator{T}}) where {S,T} = LowRankOperator{S,T}(U,V)
function LowRankOperator(U::Vector{VFun{S,T1}}, V::Vector{<:Operator}) where {S,T1}
    T2 = eltype(eltype(v))
    T = promote_type(T1,T2)
    LowRankOperator(strictconvert(Vector{VFun{S,T}},U), map(Operator{T}, V))
end

LowRankOperator(A::Fun, B::Operator) = LowRankOperator([A], [B])


function convert(::Type{Operator{T}},L::LowRankOperator{S}) where {S,T}
    L isa Operator{T} && return L
    LowRankOperator{S,T}(strictconvert(Vector{VFun{S,T}},L.U),
                         map(Operator{T}, L.V))
end


datasize(L::LowRankOperator,k) = datasize(L)[k]
datasize(L::LowRankOperator) = (mapreduce(ncoefficients,max,L.U), mapreduce(a -> bandwidth(a,1) + bandwidth(a,2)+1,max,L.V))
bandwidths(L::LowRankOperator) = datasize(L) .- 1

domainspace(L::LowRankOperator) = domainspace(first(L.V))
rangespace(L::LowRankOperator) = space(first(L.U))
promoterangespace(L::LowRankOperator,sp::Space) = LowRankOperator(map(u->Fun(u,sp),L.U),L.V)
promotedomainspace(L::LowRankOperator,sp::Space) = LowRankOperator(L.U,map(v->promotedomainspace(v,sp),L.V))

function getindex(L::LowRankOperator, k::Integer,j::Integer)
    ret=zero(eltype(L))
    for (p, LUp) in enumerate(L.U)
        if k ≤ ncoefficients(LUp)
            ret += coefficient(LUp, k) * L.V[p][j]
        end
    end
    ret
end



rank(L::LowRankOperator) = length(L.U)


-(L::LowRankOperator) = LowRankOperator(-L.U,L.V)

*(L::LowRankOperator,f::Fun) = sum(map((u,v)->u*(v*f),L.U,L.V))


*(A::LowRankOperator,B::LowRankOperator) = LowRankOperator(transpose(A.V*B.transpose(U))*A.U,B.V)
# avoid ambiguituy
for TYP in (:TimesOperator,:PlusOperator,:Conversion,:Operator)
    @eval *(L::LowRankOperator,B::$TYP) = LowRankOperator(L.U,map(v->v*B,L.V))
    @eval *(B::$TYP,L::LowRankOperator) = LowRankOperator(map(u->B*u,L.U),L.V)
end

+(A::LowRankOperator,B::LowRankOperator) = LowRankOperator([A.U;B.U],[A.V;B.V])
-(A::LowRankOperator,B::LowRankOperator) = LowRankOperator([A.U;-B.U],[A.V;B.V])
