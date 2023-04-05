"""
    ArraySpace(s::Space,dims...)

is used to represent array-valued expansions in a space `s`.  The
coefficients are of each entry are interlaced.

For example,
```julia
f = Fun(x->[exp(x),sin(x)],-1..1)
space(f) == ArraySpace(Chebyshev(),2)
```
"""
struct ArraySpace{S,n,DD,RR,A<:AbstractArray{S,n}} <: DirectSumSpace{NTuple{n,S},DD,Array{RR,n}}
     spaces::A
end

const VectorSpace{S,DD,RR,A<:AbstractVector{S}} = ArraySpace{S,1,DD,RR,A}
const MatrixSpace{S,DD,RR,A<:AbstractMatrix{S}} = ArraySpace{S,2,DD,RR,A}

#TODO: Think through domain/domaindominsion
ArraySpace(sp::AbstractArray{SS,N}) where {D,R,SS<:Space{D,R},N} =
    ArraySpace{SS,N,D,R,typeof(sp)}(sp)
ArraySpace(sp::AbstractArray{SS,N}, f = first(sp)) where {SS<:Space,N} =
    ArraySpace{SS,N,domaintype(f),mapreduce(rangetype,promote_type,sp),typeof(sp)}(sp)
ArraySpace(S::Space,::Val{n}) where {n} = ArraySpace(@SArray fill(S,n...))
ArraySpace(S::Space,n::NTuple{N,Int}) where {N} = ArraySpace(fill(S,n))
ArraySpace(S::Space,n::Integer) = ArraySpace(S,(n,))
ArraySpace(S::Space,n,m) = ArraySpace(S,(n,m))
ArraySpace(d::Domain,n...) = ArraySpace(Space(d),n...)

Space(sp::AbstractArray{<:Space}) = ArraySpace(sp)
convert(::Type{A}, sp::ArraySpace) where {A<:Array} = convert(A, sp.spaces)::A
(::Type{A})(sp::ArraySpace) where {A<:Array} = A(sp.spaces)


BlockInterlacer(sp::ArraySpace) = BlockInterlacer(map(blocklengths, Tuple(sp.spaces)))
interlacer(sp::ArraySpace) = BlockInterlacer(sp)

for OP in (:length,:firstindex,:lastindex,:size)
    @eval begin
        $OP(S::ArraySpace) = $OP(components(S))
        $OP(f::Fun{<:ArraySpace}) = $OP(space(f))
    end
end

for OP in (:getindex,:iterate,:stride,:size,:lastindex,:firstindex)
    @eval $OP(S::ArraySpace,k) = $OP(components(S),k)
end

iterate(S::ArraySpace) = iterate(components(S))
getindex(S::ArraySpace, kr::AbstractVector) = ArraySpace(components(S)[kr])

#support tuple set

stride(f::Fun{<:ArraySpace},k) = stride(space(f),k)

getindex(f::ArraySpace,k...) = Space(component(f,k...))
iterate(f::Fun{<:ArraySpace}) = iterate(components(f))


Base.reshape(AS::ArraySpace,k...) = ArraySpace(reshape(AS.spaces,k...))
dimension(AS::ArraySpace) = mapreduce(dimension,+,AS.spaces,init=0)

# TODO: union domain
domain(AS::ArraySpace) = domain(AS.spaces[1])
setdomain(A::ArraySpace,d::Domain) = ArraySpace(map(sp->setdomain(sp,d),A.spaces))



# support for Array of PiecewiseSpace


## transforms

#TODO: rework for different spaces
points(d::ArraySpace,n) = points(d.spaces[1],n)


transform(AS::ArraySpace{SS,1},vals::AbstractVector{Vector{V}}) where {SS,V} =
    transform(AS,transpose(hcat(vals...)))


function transform(AS::ArraySpace{SS,1,T},M::AbstractArray{V,2}) where {SS,T,V<:Number}
    n=length(AS)

    @assert size(M,2) == n
    plan = plan_transform(AS.spaces[1],M[:,1])
    cfs=Vector{V}[plan*M[:,k]  for k=1:size(M,2)]

    interlace(cfs,AS)
end

# transform of array is same order as vectorizing and then transforming
transform(AS::ArraySpace{SS,n},vals::AbstractVector{Array{V,n}}) where {SS,n,V} =
    transform(vec(AS),map(vec,vals))
transform(AS::VectorSpace{SS},vals::AbstractVector{AV}) where {SS,AV<:AbstractVector} =
    transform(AS,map(Vector,vals))
transform(AS::VectorSpace{SS},vals::AbstractVector{SVector{V,n}}) where {SS,n,V} =
    transform(AS,map(Vector,vals))

function itransform(AS::VectorSpace,cfs::AbstractVector)
    vf = vec(Fun(AS, cfs))
    n = maximum(ncoefficients, vf)
    vcat.(values.(pad!.(vf, n))...)
end


Base.vec(AS::ArraySpace) = ArraySpace(vec(AS.spaces))
Base.vec(f::Fun{<:ArraySpace}) = [f[j] for j=1:length(f.space)]

repeat(A::ArraySpace,n,m) = ArraySpace(repeat(A.spaces,n,m))

component(A::MatrixSpace,k::Integer,j::Integer) = A.spaces[k,j]

Base.getindex(f::Fun{DSS},k::Integer) where {DSS<:ArraySpace} = component(f,k)


Base.getindex(f::Fun{<:MatrixSpace},k::Integer,j::Integer) =
    f[k+stride(f,2)*(j-1)]

Base.getindex(f::Fun{DSS},kj::CartesianIndex{1}) where {DSS<:ArraySpace} = f[kj[1]]
Base.getindex(f::Fun{DSS},kj::CartesianIndex{2}) where {DSS<:ArraySpace} = f[kj[1],kj[2]]


function Fun(A::AbstractMatrix{<:Fun{<:VectorSpace{S},V,VV}}) where {S,V,VV}
    @assert size(A,1)==1

    M = Matrix{Fun{S,V,VV}}(undef, length(space(A[1])),size(A,2))
    for k=1:size(A,2)
        M[:,k]=vec(A[k])
    end
    Fun(M)
end

# Fun{SS,n}(v::AbstractArray{Any,n},sp::ArraySpace{SS,n}) = Fun(map((f,s)->Fun(f,s),v,sp))


# convert a vector to a Fun with ArraySpace



function Fun(v::AbstractVector,sp::Space{D,R}) where {D,R<:AbstractVector}
    if size(v) ≠ size(sp)
        throw(DimensionMismatch("Cannot convert $v to a Fun in space $sp"))
    end
    Fun(map(Fun,v,components(sp)))
end

Fun(v::AbstractArray{TT,n},sp::Space{D,R}) where {D,R<:AbstractArray{SS,n}} where {TT,SS,n} =
    reshape(Fun(vec(v),vec(sp)),size(sp))


coefficients(v::AbstractArray{TT,n},sp::ArraySpace{SS,n}) where {TT,SS,n} = coefficients(Fun(v,sp))


for (OPrule,OP) in ((:conversion_rule,:conversion_type),(:maxspace_rule,:maxspace),
                        (:union_rule,:union))
    # ArraySpace doesn't allow reordering
    @eval function $OPrule(S1::ArraySpace,S2::ArraySpace)
        sps = map($OP,S1.spaces,S2.spaces)
        for s in sps
            if isa(s,NoSpace)
                return NoSpace()
            end
        end
        ArraySpace(sps)
    end
end

## routines

spacescompatible(AS::ArraySpace,BS::ArraySpace) =
    size(AS) == size(BS) && all(((x,y),) -> spacescompatible(x,y), zip(AS.spaces,BS.spaces))
canonicalspace(AS::ArraySpace) = ArraySpace(canonicalspace.(AS.spaces))
evaluate(f::AbstractVector,S::ArraySpace,x) = map(g->g(x),Fun(S,f))

==(A::ArraySpace, B::ArraySpace) = size(A) == size(B) && all(((x,y),) -> x==y, zip(A.spaces, B.spaces))

## choosedomainspace

function choosedomainspace(A::VectorInterlaceOperator, sp::ArraySpace)
    # this ensures correct dispatch for union
    sps = filter(!isambiguous, map(choosedomainspace,A.ops,sp.spaces))
    if isempty(sps)
        UnsetSpace()
    else
        reduce(union, sps)
    end
end


Base.reshape(f::Fun{AS},k...) where {AS<:ArraySpace} = Fun(reshape(space(f),k...),f.coefficients)

Base.diff(f::Fun{AS,T},n...) where {AS<:ArraySpace,T} = Fun(diff(Array(f),n...))

## conversion

function coefficients(f::AbstractVector, a::VectorSpace, b::VectorSpace)
    if size(a) ≠ size(b)
        throw(DimensionMismatch("dimensions must match"))
    end
    interlace(map(coefficients,Fun(a,f),b),b)
end


coefficients(Q::AbstractVector{F},rs::VectorSpace) where {F<:Fun} =
    interlace(map(coefficients,Q,rs),rs)

coefficients(Q::AbstractVector, rs::VectorSpace) = coefficients(Fun.(Q), rs)


Fun(f::AbstractVector{FF},d::VectorSpace) where {FF<:Fun} = Fun(d,coefficients(f,d))
Fun(f::AbstractMatrix{FF},d::MatrixSpace) where {FF<:Fun} = Fun(d,coefficients(f,d))





## constructor



# columns are coefficients
function Fun(M::AbstractMatrix{<:Number},sp::MatrixSpace)
    if size(M) ≠ size(sp)
        throw(DimensionMismatch("size of array $(size(M)) doesn't match that of the space $(size(sp))"))
    end
    Fun(map(Fun, M, sp.spaces))
end

Fun(M::UniformScaling,sp::MatrixSpace) = Fun(Matrix(M,size(sp)),sp)



ones(::Type{T},A::ArraySpace) where {T<:Number} = Fun(ones.(T,A.spaces))
ones(A::ArraySpace) = ones(Float64, A)


## EuclideanSpace

const EuclideanSpace{RR} = VectorSpace{ConstantSpace{AnyDomain},AnyDomain,RR}
EuclideanSpace(n::Integer) = ArraySpace(ConstantSpace(Float64),n)




## support pieces

npieces(f::Fun{<:ArraySpace}) = npieces(f[1])
piece(f::Fun{<:ArraySpace}, k) = Fun(piece.(Array(f),k))
pieces(f::Fun{<:ArraySpace}) = [piece(f,k) for k=1:npieces(f)]



## TODO: This is a hack to get tests working

fromcanonical(d::ProductDomain, f::Fun{<:ArraySpace}) = vcat(fromcanonical.(factors(d), vec(f))...)

function coefficients(f::AbstractVector,sp::ArraySpace{<:ConstantSpace{AnyDomain}},ts::TensorSpace{SV,D,R}) where {SV,D<:EuclideanDomain{2},R}
    @assert length(ts.spaces) == 2

    if ts.spaces[1] isa ArraySpace
        coefficients(f, sp, ts.spaces[1])
    elseif ts.spaces[2] isa ArraySpace
        coefficients(f, sp, ts.spaces[2])
    else
        error("Cannot convert coefficients from $sp to $ts")
    end
end



ArraySpace(sp::TensorSpace{Tuple{S1,S2}}) where {S1<:Space{D,R},S2} where {D,R<:AbstractArray} =
    ArraySpace(map(a -> a ⊗ sp.spaces[2], sp.spaces[1]))

ArraySpace(sp::TensorSpace{Tuple{S1,S2}},k...) where {S1,S2<:Space{D,R}} where {D,R<:AbstractArray} =
    ArraySpace(map(a -> sp.spaces[1] ⊗ a, sp.spaces[2]))

function coefficients(f::AbstractVector, a::VectorSpace, b::TensorSpace{Tuple{S1,S2},<:EuclideanDomain{2}}) where {S1<:Space{D,R},S2} where {D,R<:AbstractArray}
    if size(a) ≠ size(b)
        throw(DimensionMismatch("dimensions must match"))
    end
    interlace(map(coefficients,Fun(a,f),b),ArraySpace(b))
end

function coefficients(f::AbstractVector, a::VectorSpace, b::TensorSpace{Tuple{S1,S2},<:EuclideanDomain{2}}) where {S1,S2<:Space{D,R}} where {D,R<:AbstractArray}
    if size(a) ≠ size(b)
        throw(DimensionMismatch("dimensions must match"))
    end
    interlace(map(coefficients,Fun(a,f),b),ArraySpace(b))
end
