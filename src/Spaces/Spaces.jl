

include("ConstantSpace.jl")
include("SumSpace.jl")
include("ArraySpace.jl")
include("ProductSpaceOperators.jl")
include("SubSpace.jl")
include("QuotientSpace.jl")


⊕(A::Space,B::Space)::Any = domainscompatible(A,B) ? SumSpace(A,B) : PiecewiseSpace(A,B)
⊕(f::Fun,g::Fun) = Fun(space(f) ⊕ space(g), interlace(coefficients(f),coefficients(g)))

⊕(f::Fun,g::Fun,h::Fun...) = ⊕((f ⊕ g), h...)

+(A::Space,B::Space) = A ⊕ B




#split the cfs into component spaces
function coefficients(cfs::AbstractVector,A::SumSpace,B::SumSpace)
    if spacescompatible(A,B)
        cfs
    else
        mapreduce(f->Fun(f,B),+,components(Fun(A,cfs))).coefficients
    end
end
function coefficients(cfs::AbstractVector,A::PiecewiseSpace,B::PiecewiseSpace)
    if spacescompatible(A,B)
        cfs
    else
        mapreduce(f->Fun(f,B),+,components(Fun(A,cfs))).coefficients
    end
end


# spread a single space into a sum space by placing
# its coefficients depending on k
function interlacewithzeros(cfs::AbstractVector,k,it)
    n = length(cfs)

    ret = Array{eltype(cfs)}(undef, 0)
    n == 0 && return ret

    for (K,j) in it
        if K == k
            push!(ret,cfs[j])
            n == j && break
        else
            push!(ret,0)
        end
    end

    ret
end

interlacewithzeros(cfs::AbstractVector,k,B::DirectSumSpace) = interlacewithzeros(cfs,k,interlacer(B))

function sumspacecoefficients(cfsin::AbstractVector,A::Space,B::SumSpace)
    m=length(components(B))

    for k=1:m
        if isconvertible(A,component(B,k))
            cfs = coefficients(cfsin,A,component(B,k))
            return interlacewithzeros(cfs,k,B)
        end
    end

    defaultcoefficients(cfsin,A,B)
end

function sumspacecoefficients(cfsin::AbstractVector,A::Space,B::PiecewiseSpace)
    m=length(components(B))

    for k=1:m
        if domain(component(B,k)) == domain(A) && isconvertible(A,component(B,k))
            cfs = coefficients(cfsin,A,component(B,k))
            return interlacewithzeros(cfs,k,B)
        end
    end

    defaultcoefficients(cfsin,A,B)
end

for TYP in (:SumSpace,:PiecewiseSpace), ATYP in (:ConstantSpace,:(ConstantSpace{<:Domain{<:Number}}),:Space)
    @eval coefficients(cfsin::AbstractVector,A::$ATYP,B::$TYP) = sumspacecoefficients(cfsin,A,B)
end


## LowRank Constructors

# convert a vector of functionals and an operator to a LowRnakPertOperator
# the rangespace is a DirectSumSpace specified by ST of the input rangespaces
# the default is a  ArraySpace, but support is there for PiecewiseSpace
# for bcs
for TYP in (:PiecewiseSpace,:VectorSpace)
    @eval function LowRankPertOperator(A::AbstractVector{OT},::Type{$TYP}) where OT<:Operator
        A=promotedomainspace(A)
        for k=1:length(A)-1
            @assert isafunctional(A[k])
        end
        @assert isbanded(A[end])
        L=LowRankOperator(A[1:end-1],$TYP)
        # add zero functionals to shift down
        BB=[fill(ZeroOperator(domainspace(BB),ConstantSpace()),length(A)-1);A[end]]
        S=InterlaceOperator(BB,domainspace(BB),$TYP(map(rangespace,A)))
        L+S
    end
end

LowRankPertOperator(A::AbstractVector{OT}) where {OT<:Operator}=LowRankPertOperator(A,VectorSpace)

function LowRankOperator(Bin::AbstractVector{FT},::Type{PiecewiseSpace}) where FT<:Operator
    B=promotedomainspace(Bin)
    rsp=PiecewiseSpace(map(rangespace,B))
    LowRankOperator(
        VFun{typeof(rsp),Float64}[Fun(rsp,[zeros(k-1);1]) for k=1:length(B)],
        B)
end

function LowRankOperator(Bin::AbstractVector{FT},::Type{VectorSpace}) where FT<:Operator
    B=promotedomainspace(Bin)
    rsp=Space([map(rangespace,B);ZeroSpace()])  #TODO: Why the hack?
    LowRankOperator(
        VFun{typeof(rsp),Float64}[Fun(rsp,[zeros(k-1);1]) for k=1:length(B)],
        B)
end



LowRankOperator(Bin::AbstractVector{FT}) where {FT<:Operator} = LowRankOperator(Bin,VectorSpace)


"""
 WeightSpace represents a space that weights another space.
 Overload weight(S,x).
"""
abstract type WeightSpace{S,DD,RR} <: Space{DD,RR} end


domain(S::WeightSpace) = domain(S.space)
points(sp::WeightSpace,n) = points(sp.space,n)

weight(S::WeightSpace, x...) = error("Override `weight(::$S, $x)`")


struct WeightSpacePlan{S,P,T,V}
    space::S
    plan::P
    points::Vector{T}
    weights::Vector{V}
end

struct IWeightSpacePlan{S,P,T,V}
    space::S
    plan::P
    points::Vector{T}
    weights::Vector{V}
end

function plan_transform(S::WeightSpace,vals::AbstractVector)
    pts=points(S,length(vals))
    WeightSpacePlan(S,plan_transform(S.space,vals),pts,weight.(Ref(S),pts))
end

function plan_itransform(S::WeightSpace,vals::AbstractVector)
    pts=points(S,length(vals))
    IWeightSpacePlan(S,plan_itransform(S.space,vals),pts,weight.(Ref(S),pts))
end

*(P::WeightSpacePlan,vals::AbstractVector) = P.plan*(vals./P.weights)
*(P::IWeightSpacePlan,cfs::AbstractVector) = P.weights.*(P.plan*cfs)


# used for ProductFun
transform(sp::WeightSpace,vals::AbstractVector,plan::WeightSpacePlan) =
    transform(sp.space,vals./(sp==plan.space ? plan.weights : weight.(sp,plan.points)),plan.plan)
itransform(sp::WeightSpace,cfs::AbstractVector,plan::WeightSpacePlan) =
    itransform(sp.space,cfs,plan.plan).*(sp==plan.space ? plan.weights : weight.(sp,plan.points))



evaluate(f::AbstractVector,S::WeightSpace,x) = weight(S,x)*evaluate(f,S.space,x)

function evaluate(f::AbstractVector,S::WeightSpace,x...)
    fv=evaluate(f,S.space,x...)
    weight(S,x...).*fv
end

# recurrence is inhereted
for FUNC in (:recα,:recβ,:recγ)
    @eval $FUNC(T,ws::WeightSpace,k) = $FUNC(T,ws.space,k)
end

include("HeavisideSpace.jl")
include("DiracSpace.jl")