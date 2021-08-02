include("ProductDomain.jl")


## boundary

const RectDomain = Union{DomainSets.FixedIntervalProduct{2}, DomainSets.Rectangle, VcatDomain{2,<:Any,(1,1),<:Tuple}}

boundary(d::RectDomain) =
    PiecewiseSegment([Vec(leftendpoint(factor(d,1)),leftendpoint(factor(d,2))),
                      Vec(rightendpoint(factor(d,1)),leftendpoint(factor(d,2))),
                      Vec(rightendpoint(factor(d,1)),rightendpoint(factor(d,2))),
                      Vec(leftendpoint(factor(d,1)),rightendpoint(factor(d,2))),
                      Vec(leftendpoint(factor(d,1)),leftendpoint(factor(d,2)))])


## Union
function Base.join(p1::AbstractVector{IT},p2::AbstractVector{IT}) where IT<:IntervalOrSegment
    for k=length(p1):-1:1,j=length(p2):-1:1
        if p1[k]==reverse(p2[j])
            deleteat!(p1,k)
            deleteat!(p2,j)
            break
        end
    end
    [p1;p2]
end

function boundary(d::UnionDomain{Tuple{PD1,PD2}}) where {PD1<:ProductDomain,PD2<:ProductDomain}
    bnd=map(d->components(∂(d)),d.domains)
    PiecewiseSegment(reduce(join,bnd))
end
