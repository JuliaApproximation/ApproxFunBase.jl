struct ContinuousSpace{T,R,P<:PiecewiseSegment{T}} <: Space{P,R}
    domain::P
end

ContinuousSpace(d::PiecewiseSegment{T}) where {T} =
    ContinuousSpace{T,real(eltype(T)),typeof(d)}(d)


Space(d::PiecewiseSegment) = ContinuousSpace(d)

isperiodic(C::ContinuousSpace) = isperiodic(domain(C))

spacescompatible(a::ContinuousSpace, b::ContinuousSpace) = domainscompatible(a,b)

plan_transform(sp::ContinuousSpace, vals::AbstractVector) =
    TransformPlan{eltype(vals),typeof(sp),false,Nothing}(sp,nothing)

function *(P::TransformPlan{T,<:ContinuousSpace,false}, vals::AbstractVector{T}) where {T}
    S = P.space
    n=length(vals)
    d=domain(S)
    K=ncomponents(d)
    k=div(n,K)

    PT=promote_type(prectype(d),eltype(vals))
    if k==0
        vals
    elseif isperiodic(d)
        ret=Array{PT}(undef, max(K,n-K))
        r=n-K*k

        for j=1:r
            cfs=transform(component(S,j), vals[(j-1)*(k+1)+1:j*(k+1)])
            if j==1
                ret[1]=cfs[1]-cfs[2]
                ret[2]=cfs[1]+cfs[2]
            elseif j < K
                ret[j+1]=cfs[1]+cfs[2]
            end
            ret[K+j:K:end]=cfs[3:end]
        end

        for j=r+1:K
            cfs=transform(component(S,j), vals[r*(k+1)+(j-r-1)*k+1:r*(k+1)+(j-r)*k])
            if length(cfs)==1 && j <K
                ret[j+1]=cfs[1]
            elseif j==1
                ret[1]=cfs[1]-cfs[2]
                ret[2]=cfs[1]+cfs[2]
            elseif j < K
                ret[j+1]=cfs[1]+cfs[2]
            end
            ret[K+j:K:end]=cfs[3:end]
        end

        ret
    else
        ret=Array{PT}(undef, n-K+1)
        r=n-K*k

        for j=1:r
            cfs=transform(component(S,j), vals[(j-1)*(k+1)+1:j*(k+1)])
            if j==1
                ret[1]=cfs[1]-cfs[2]
            end

            ret[j+1]=cfs[1]+cfs[2]
            ret[K+j+1:K:end]=cfs[3:end]
        end

        for j=r+1:K
            cfs=transform(component(S,j), vals[r*(k+1)+(j-r-1)*k+1:r*(k+1)+(j-r)*k])

            if length(cfs) ≤ 1
                ret .= cfs
            else
                if j==1
                    ret[1]=cfs[1]-cfs[2]
                end
                ret[j+1]=cfs[1]+cfs[2]
                ret[K+j+1:K:end]=cfs[3:end]
            end
        end

        ret
    end
end

canonicalspace(S::ContinuousSpace) = PiecewiseSpace(components(S))
convert(::Type{PiecewiseSpace}, S::ContinuousSpace) = canonicalspace(S)

blocklengths(C::ContinuousSpace) = Fill(ncomponents(C.domain),∞)

block(C::ContinuousSpace,k) = Block((k-1)÷ncomponents(C.domain)+1)


## components

components(f::Fun{<:ContinuousSpace},j::Integer) = components(Fun(f,canonicalspace(f)),j)
components(f::Fun{<:ContinuousSpace}) = components(Fun(f,canonicalspace(space(f))))


function points(f::Fun{<:ContinuousSpace})
    n=ncoefficients(f)
    d=domain(f)
    K=ncomponents(d)

    m=isperiodic(d) ? max(K,n+2K-1) : n+K
    points(f.space,m)
end

## Conversion

coefficients(cfsin::AbstractVector,A::ContinuousSpace,B::PiecewiseSpace) =
    defaultcoefficients(cfsin,A,B)

coefficients(cfsin::AbstractVector,A::PiecewiseSpace,B::ContinuousSpace) =
    default_Fun(Fun(A,cfsin),B).coefficients

coefficients(cfsin::AbstractVector,A::ContinuousSpace,B::ContinuousSpace) =
    default_Fun(Fun(A,cfsin),B).coefficients

union_rule(A::PiecewiseSpace, B::ContinuousSpace) = union(A, strictconvert(PiecewiseSpace, B))
union_rule(A::ConstantSpace, B::ContinuousSpace) = B

function approx_union(a::AbstractVector{T}, b::AbstractVector{V}) where {T,V}
    ret = sort!(union(a,b))
    for k in length(ret)-1:-1:1
        isapprox(ret[k] , ret[k+1]; atol=10eps()) && deleteat!(ret, k+1)
    end
    ret
end


function union_rule(A::ContinuousSpace{<:Real}, B::ContinuousSpace{<:Real})
    p_A,p_B = domain(A).points, domain(B).points
    a,b = minimum(p_A),  maximum(p_A)
    c,d = minimum(p_B),  maximum(p_B)
    @assert !isempty((a..b) ∩ (c..d))
    ContinuousSpace(PiecewiseSegment(approx_union(p_A, p_B)))
end

function integrate(f::Fun{<:ContinuousSpace})
    cs = [cumsum(x) for x in components(f)]
    for k=1:length(cs)-1
        cs[k+1] += last(cs[k])
    end
    Fun(Fun(cs, PiecewiseSpace), space(f))
end

cumsum(f::Fun{<:ContinuousSpace}) = integrate(f)
