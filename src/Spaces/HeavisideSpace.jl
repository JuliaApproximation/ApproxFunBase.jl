# SplineSpace represents a Spline, right now piecewise constant HeavisideSpace is only implemented case
struct SplineSpace{order,T,R,P<:PiecewiseSegment} <: Space{P,R}
    domain::P
end

function SplineSpace{order,T,R,P}(S::SplineSpace) where {order,T,R,P<:PiecewiseSegment}
    SplineSpace{order,T,R,P}(strictconvert(P, S.domain))
end

SplineSpace{m,T,R}(d::PiecewiseSegment{T}) where {m,T,R} = SplineSpace{m,T,R,typeof(d)}(d)

SplineSpace{m,T}(d::PiecewiseSegment{T}) where {m,T} = SplineSpace{m,T,real(eltype(T))}(d)
SplineSpace{m,T}(d::AbstractVector) where {m,T} = SplineSpace{m}(PiecewiseSegment(sort(d)))

SplineSpace{m}(d::PiecewiseSegment{T}) where {m,T} = SplineSpace{m,T,real(eltype(T))}(d)
SplineSpace{m}(d::AbstractVector) where {m} = SplineSpace{m}(PiecewiseSegment(sort(d)))

const HeavisideSpace{T,R,P<:PiecewiseSegment} = SplineSpace{0,T,R,P}
dimension(h::SplineSpace{λ}) where {λ} = length(h.domain.points)+λ-1

convert(::Type{HeavisideSpace},d::PiecewiseSegment) = HeavisideSpace{eltype(d)}(d)

convert(::Type{HeavisideSpace},d::AbstractVector) =
    HeavisideSpace(PiecewiseSegment(sort(d)))

spacescompatible(a::SplineSpace{λ},b::SplineSpace{λ}) where {λ} = domainscompatible(a,b)


function evaluate(c::AbstractVector{T}, s::HeavisideSpace{<:Real}, x::Real) where T
    p = domain(s).points
    for k=1:length(c)
        if p[k] ≤ x ≤ p[k+1]
            return c[k]
        end
    end
    return zero(T)
end


function evaluate(c::AbstractVector{T}, s::SplineSpace{1,<:Real}, x::Real) where T
    p = domain(f).points
    c = f.coefficients
    for k=1:length(p)-1
        if p[k] ≤ x ≤ p[k+1]
            return (x-p[k])*c[k+1]/(p[k+1]-p[k]) + (p[k+1]-x)*c[k]/(p[k+1]-p[k])
        end
    end
    return zero(T)
end


function points(sp::HeavisideSpace,n)
    x=sp.domain.points
    (x[1:end-1] + diff(x)/2)[1:n]
end

points(sp::SplineSpace{1},n) = sp.domain.points[1:n]

for λ = [0,1]
    @eval begin
        function transform(S::SplineSpace{$λ},vals::AbstractVector,plan...)
            @assert length(vals) ≤ dimension(S)
            vals
        end
        itransform(S::SplineSpace{$λ},cfs::AbstractVector,plan...) = pad(cfs,dimension(S))
    end
end


bandwidths(D::ConcreteDerivative{H}) where {H<:HeavisideSpace} = (0,1)
rangespace(D::ConcreteDerivative{H}) where {H<:HeavisideSpace} = DiracSpace(domain(D).points[2:end-1])

function getindex(D::ConcreteDerivative{H,<:Any,T},k::Integer,j::Integer) where {H<:HeavisideSpace,T}
    if k==j
        -one(T)
    elseif j==k+1
        one(T)
    else
        zero(T)
    end
end

Base.sum(f::Fun{HS}) where {HS<:HeavisideSpace} = dotu(f.coefficients,diff(space(f).domain.points))
function Base.sum(f::Fun{SplineSpace{1,T,R}}) where {T,R}
    vals=pad(f.coefficients,dimension(space(f)))
    dfs=diff(space(f).domain.points)
    ret=vals[1]*dfs[1]/2
    for k=2:length(vals)-1
        ret+=vals[k]*(dfs[k]+dfs[k+1])/2
    end
    ret+=vals[end]*dfs[end]/2
    ret
end

#diffentiate HeavisideSpace
function differentiate(f::Fun{<:HeavisideSpace})
    dp=domain(f).points
    cfs=f.coefficients
    diff=0.0
    for n=1:length(cfs)
        diff=diff+cfs[n]*(DiracDelta(dp[n])-DiracDelta(dp[n+1]))
    end
    return diff
end

#Derivative Operator for HeavisideSpace
function Derivative(H::HeavisideSpace, k::Int)
    @assert k == 1
    ConcreteDerivative(H)
end



differentiate(f::Fun{<:SplineSpace{1}}) =
    Fun(HeavisideSpace(space(f).domain),
        diff(pad(f.coefficients,dimension(space(f))))./diff(space(f).domain.points))

integrate(f::Fun{<:HeavisideSpace{T,R}}) where {T,R} =
    Fun(SplineSpace{1,T,R}(space(f).domain),
        [0;cumsum(f.coefficients).*diff(space(f).domain.points)])
