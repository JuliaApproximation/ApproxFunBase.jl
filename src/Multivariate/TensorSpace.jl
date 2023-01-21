
export TensorSpace, ⊗, ProductSpace, factor, factors, nfactors

#  SV is a tuple of d spaces
abstract type AbstractProductSpace{SV,DD,RR} <: Space{DD,RR} end


spacetype(::AbstractProductSpace{SV},k) where {SV} = SV.parameters[k]


##### Tensorizer
# This gives the map from coefficients to the
# tensor entry of a tensor product of d spaces
# findfirst is overriden to get efficient inverse
# blocklengths is a tuple of block lengths, e.g., Chebyshev()^2
# would be Tensorizer((1:∞,1:∞))
# ConstantSpace() ⊗ Chebyshev()
# would be Tensorizer((1:1,1:∞))
# and Chebyshev() ⊗ ArraySpace([Chebyshev(),Chebyshev()])
# would be Tensorizer((1:∞,2:2:∞))


struct Tensorizer{DMS<:Tuple}
    blocks::DMS
end

const Tensorizer2D{AA, BB} = Tensorizer{Tuple{AA, BB}}
const TrivialTensorizer{d} = Tensorizer{NTuple{d,Ones{Int,1,Tuple{OneToInf{Int}}}}}

eltype(::Type{<:Tensorizer{<:Tuple{Vararg{Any,N}}}}) where {N} = NTuple{N,Int}
dimensions(a::Tensorizer) = map(sum,a.blocks)
Base.length(a::Tensorizer) = reduce(*, dimensions(a)) # easier type-inference than mapreduce

Base.keys(a::Tensorizer) = oneto(length(a))

function start(a::TrivialTensorizer{d}) where {d}
    # ((block_dim_1, block_dim_2,...), (itaration_number, iterator, iterator_state)), (itemssofar, length)
    block = ntuple(one, d)
    return (block, (0, nothing, nothing)), (0,length(a))
end

function next(a::TrivialTensorizer{d}, iterator_tuple) where {d}
    (block, (j, iterator, iter_state)), (i,tot) = iterator_tuple

    @inline function check_block_finished(j, iterator, block)
        if iterator === nothing
            return true
        end
        # there are N-1 over d-1 combinations in a block
        amount_combinations_block = binomial(sum(block)-1, d-1)
        # check if all combinations have been iterated over
        amount_combinations_block <= j
    end

    ret = reverse(block)

    if check_block_finished(j, iterator, block)   # end of new block
        # set up iterator for new block
        current_sum = sum(block)
        iterator = multiexponents(d, current_sum+1-d)
        iter_state = nothing
        j = 0
    end

    # increase block, or initialize new block
    _res, iter_state = iterate(iterator, iter_state)
    res = Tuple(SVector{d}(_res))
    block = res.+1
    j = j+1

    ret, ((block, (j, iterator, iter_state)), (i,tot))
end


function done(a::TrivialTensorizer, iterator_tuple)
    i, tot = last(iterator_tuple)
    return i ≥ tot
end


# (blockrow,blockcol), (subrow,subcol), (rowshift,colshift), (numblockrows,numblockcols), (itemssofar, length)
start(a::Tensorizer2D) = _start(a)
start(a::TrivialTensorizer{2}) = _start(a)

_start(a) = (1,1), (1,1), (0,0), (a.blocks[1][1],a.blocks[2][1]), (0,length(a))

next(a::Tensorizer2D, state) = _next(a, state)
next(a::TrivialTensorizer{2}, state) = _next(a, state)

function _next(a, ((K,J), (k,j), (rsh,csh), (n,m), (i,tot)))
    ret = k+rsh,j+csh
    if k==n && j==m  # end of block
        if J == 1 || K == length(a.blocks[1])   # end of new block
            B = K+J # next block
            J = min(B, length(a.blocks[2]))::Int  # don't go past new block
            K = B-J+1   # K+J-1 == B
        else
            K,J = K+1,J-1
        end
        k = j = 1
        if i+1 < tot # not done yet
            n,m = a.blocks[1][K], a.blocks[2][J]
            rsh,csh = sum(a.blocks[1][1:K-1]), sum(a.blocks[2][1:J-1])
        end
    elseif k==n
        k  = 1
        j += 1
    else
        k += 1
    end
    ret, ((K,J), (k,j), (rsh,csh), (n,m), (i+1,tot))
end

done(a::Tensorizer2D, state) = _done(a, state)
done(a::TrivialTensorizer{2}, state) = _done(a, state)

_done(a, (_, _, _, _, (i,tot))) = i ≥ tot

iterate(a::Tensorizer) = next(a, start(a))
function iterate(a::Tensorizer, st)
    done(a,st) && return nothing
    next(a, st)
end


cache(a::Tensorizer) = CachedIterator(a)

function Base.findfirst(::TrivialTensorizer{2},kj::Tuple{Int,Int})
    k,j=kj
    if k > 0 && j > 0
        n=k+j-2
        (n*(n+1))÷2+k
    else
        nothing
    end
end
function Base.findfirst(sp::Tensorizer{<:NTuple{2,Ones{Int}}}, kj::NTuple{2,Int})
    k,j=kj

    len1, len2 = length(sp.blocks[1]), length(sp.blocks[2])
    if 0 < k <= len1 && 0 < j <= len2
        kb1 = k-1
        jb1 = j-1
        nb=kb1+jb1+1
        # Fully filled blocks, that go from (1,n) to (n,1)
        nb_full = min(nb-1, len2)
        ind = nb_full*(nb_full+1)÷2
        # Number of partially filled blocks, that is blocks where `a` in (`a`,`b`) doesn't start at 1
        # This happens when `b` in a block starts from `length(sp.blocks[2])` and `a` starts from > 1
        nb_part = nb - nb_full - 1
        if nb_part > 0
            ind -= nb_part * (1 - 2nb_full + nb_part) ÷ 2
        end
        sum12 = nb + 1 # a + b where the element is (a,b)
        start2 = min(nb, len2) # the second element is bound by the number of blocks
        nel_block = start2 - j + 1 # the second index decreases
        ind += nel_block
    else
        nothing
    end
end

# which block of the tensor
# equivalent to sum of indices -1

# block(it::Tensorizer,k) = Block(sum(it[k])-length(it.blocks)+1)
block(ci::CachedIterator{T,TrivialTensorizer{2}},k::Int) where {T} =
    Block(k == 0 ? 0 : sum(ci[k])-length(ci.iterator.blocks)+1)

block(::TrivialTensorizer{2},n::Int) =
    Block(floor(Integer,sqrt(2n) + 1/2))

function block(::TrivialTensorizer{d},n::Int) where {d}
    binomial(d, d) >= n && return Block(1)
    order = 1
    while binomial(order+d, d) < n
        order *= 2
    end
    searchords = order÷2:order
    # perform a binary search
    while length(searchords) > 1
        midpt = searchords[length(searchords)÷2]
        if binomial(midpt+d, d) < n
            searchords = (midpt + 1):last(searchords)
        else
            searchords = first(searchords):midpt
        end
    end
    order = searchords[]
    return Block(order+1)
end

block(sp::Tensorizer{<:Tuple{<:AbstractFill{S},<:AbstractFill{T}}},n::Int) where {S,T} =
    Block(floor(Integer,sqrt(2floor(Integer,(n-1)/(getindex_value(sp.blocks[1])*getindex_value(sp.blocks[2])))+1) + 1/2))
_cumsum(x) = cumsum(x)
_cumsum(x::Number) = x
block(sp::Tensorizer,k::Int) = Block(findfirst(x->x≥k, _cumsum(blocklengths(sp))))
block(sp::CachedIterator,k::Int) = block(sp.iterator,k)

blocklength(it,k) = blocklengths(it)[k]
blocklength(it,k::Block) = blocklength(it,k.n[1])
blocklength(it,k::BlockRange) = blocklength(it,Int.(k))

blocklengths(::TrivialTensorizer{2}) = 1:∞



blocklengths(it::Tensorizer) = tensorblocklengths(it.blocks...)
blocklengths(it::CachedIterator) = blocklengths(it.iterator)

function getindex(it::TrivialTensorizer{2},n::Integer)
    m=Int(block(it,n))
    p=findfirst(it,(1,m))
    j=1+n-p
    j,m-j+1
end


blockstart(it,K)::Int = K==1 ? 1 : sum(blocklengths(it)[1:K-1])+1
blockstop(it,::PosInfinity) = ℵ₀
_K_sum(bl::AbstractVector, K) = sum(bl[1:K])
_K_sum(bl::Integer, K) = bl
blockstop(it, K)::Int = _K_sum(blocklengths(it), K)

blockstart(it,K::Block) = blockstart(it,K.n[1])
blockstop(it,K::Block) = blockstop(it,K.n[1])


blockrange(it,K) = blockstart(it,K):blockstop(it,K)
blockrange(it,K::BlockRange) = blockstart(it,first(K)):blockstop(it,last(K))




# convert from block, subblock to tensor
subblock2tensor(rt::TrivialTensorizer{2},K,k) =
    (k,K.n[1]-k+1)

subblock2tensor(rt::CachedIterator{II,TrivialTensorizer{2}},K,k) where {II} =
    (k,K.n[1]-k+1)


subblock2tensor(rt::CachedIterator,K,k) = rt[blockstart(rt,K)+k-1]

# tensorblocklengths gives calculates the block sizes of each tensor product
#  Tensor product degrees are taken to be the sum of the degrees
#  a degree is which block you are in


tensorblocklengths(a) = a   # a single block is not modified
tensorblocklengths(a, b) = conv(a,b)
tensorblocklengths(a,b,c,d...) = tensorblocklengths(tensorblocklengths(a,b),c,d...)


# TensorSpace
# represents the tensor product of several subspaces
"""
    TensorSpace(a::Space,b::Space)

represents a tensor product of two 1D spaces `a` and `b`.
The coefficients are interlaced in lexigraphical order.

For example, consider
```julia
Fourier()*Chebyshev()  # returns TensorSpace(Fourier(),Chebyshev())
```
This represents functions on `[-π,π) x [-1,1]`, using the Fourier basis for the first argument
and Chebyshev basis for the second argument, that is, `φ_k(x)T_j(y)`, where
```
φ_0(x) = 1,
φ_1(x) = sin x,
φ_2(x) = cos x,
φ_3(x) = sin 2x,
φ_4(x) = cos 2x
…
```
By Choosing `(k,j)` appropriately, we obtain a single basis:
```
φ_0(x)T_0(y) (= 1),
φ_0(x)T_1(y) (= y),
φ_1(x)T_0(y) (= sin x),
φ_0(x)T_2(y), …
```
"""
struct TensorSpace{SV,D,R} <:AbstractProductSpace{SV,D,R}
    spaces::SV
end

# Tensorspace of 2 univariate spaces
const TensorSpace2D{AA, BB, D,R} = TensorSpace{<:Tuple{AA, BB}, D, R} where {AA<:UnivariateSpace, BB<:UnivariateSpace}
const TensorSpaceND{d, D, R} = TensorSpace{<:NTuple{d, <:UnivariateSpace}, D, R}

tensorizer(sp::TensorSpace) = Tensorizer(map(blocklengths,sp.spaces))
blocklengths(S::TensorSpace) = tensorblocklengths(map(blocklengths,S.spaces)...)


# the evaluation is *, so the type will be the same as *
# However, this fails for some any types
tensor_eval_type(a,b) = Base.promote_op(*,a,b)
tensor_eval_type(::Type{Vector{Any}},::Type{Vector{Any}}) = Vector{Any}
tensor_eval_type(::Type{Vector{Any}},_) = Vector{Any}
tensor_eval_type(_,::Type{Vector{Any}}) = Vector{Any}

# Specialize some common cases to avoid mapreduce, which has inference issues
_typeofproddomain(sp::Tuple{Any}) = typeof(domain(sp[1]))
_typeofproddomain(sp::Tuple{Any,Any}) = typeof(domain(sp[1]) × domain(sp[2]))
_typeofproddomain(sp) = typeof(mapreduce(domain,×,sp))
TensorSpace(sp::Tuple) =
    TensorSpace{typeof(sp), _typeofproddomain(sp),
                mapreduce(rangetype,tensor_eval_type,sp)}(sp)

dimension(sp::TensorSpace) = mapreduce(dimension,*,sp.spaces)

==(A::TensorSpace{<:NTuple{N,Space}}, B::TensorSpace{<:NTuple{N,Space}}) where {N} =
        factors(A) == factors(B)

conversion_rule(a::TensorSpace{<:NTuple{2,Space}}, b::TensorSpace{<:NTuple{2,Space}}) =
    conversion_type(a.spaces[1],b.spaces[1]) ⊗ conversion_type(a.spaces[2],b.spaces[2])

maxspace_rule(a::TensorSpace{<:NTuple{2,Space}}, b::TensorSpace{<:NTuple{2,Space}}) =
    maxspace(a.spaces[1],b.spaces[1]) ⊗ maxspace(a.spaces[2],b.spaces[2])

function spacescompatible(A::TensorSpace{<:NTuple{N,Space}}, B::TensorSpace{<:NTuple{N,Space}}) where {N}
    _spacescompatible(factors(A), factors(B))
end
_spacescompatible(::Tuple{}, ::Tuple{}) = true
function _spacescompatible(A::Tuple{Space, Vararg{Space}}, B::Tuple{Space, Vararg{Space}})
    spacescompatible(A[1], B[1]) && _spacescompatible(Base.tail(A), Base.tail(B))
end

canonicalspace(T::TensorSpace) = TensorSpace(map(canonicalspace,T.spaces))


TensorSpace(A::SVector{<:Any,<:Space}) = TensorSpace(Tuple(A))
TensorSpace(A...) = TensorSpace(A)
TensorSpace(A::ProductDomain) = TensorSpace(tuple(map(Space,components(A))...))
⊗(A::TensorSpace,B::TensorSpace) = TensorSpace(A.spaces...,B.spaces...)
⊗(A::TensorSpace,B::Space) = TensorSpace(A.spaces...,B)
⊗(A::Space,B::TensorSpace) = TensorSpace(A,B.spaces...)
⊗(A::Space,B::Space) = TensorSpace(A,B)

domain(f::TensorSpace) = ×(domain.(f.spaces)...)
Space(sp::ProductDomain) = TensorSpace(sp)

setdomain(sp::TensorSpace, d::ProductDomain) = TensorSpace(setdomain.(factors(sp), factors(d)))

*(A::Space, B::Space) = A ⊗ B
@inline function _powspace(A, p)
    p >= 1 || throw(ArgumentError("exponent must be >= 1, received $p"))
    # Enumerate common cases to help with constant propagation
    p == 1 ? A :
    p == 2 ? A * A :
    p == 3 ? A * A * A :
    foldl(*, ntuple(_ -> A, p))
end
@static if VERSION >= v"1.8"
    Base.@constprop :aggressive function ^(A::Space, p::Integer)
        _powspace(A, p)
    end
else
    ^(A::Space, p::Integer) = _powspace(A, p)
end


## TODO: generalize
components(sp::TensorSpace{Tuple{S1,S2}}) where {S1<:Space{D,R},S2} where {D,R<:AbstractArray} =
    [s ⊗ sp.spaces[2] for s in components(sp.spaces[1])]

components(sp::TensorSpace{Tuple{S1,S2}}) where {S1,S2<:Space{D,R}} where {D,R<:AbstractArray} =
    [sp.spaces[1] ⊗ s for s in components(sp.spaces[2])]

Base.size(sp::TensorSpace{Tuple{S1,S2}}) where {S1<:Space{D,R},S2} where {D,R<:AbstractArray} =
    size(sp.spaces[1])

Base.size(sp::TensorSpace{Tuple{S1,S2}}) where {S1,S2<:Space{D,R}} where {D,R<:AbstractArray} =
    size(sp.spaces[2])

# TODO: Generalize to higher dimensions
getindex(sp::TensorSpace{Tuple{S1,S2}},k::Integer) where {S1<:Space{D,R},S2} where {D,R<:AbstractArray} =
    sp.spaces[1][k] ⊗ sp.spaces[2]

getindex(sp::TensorSpace{Tuple{S1,S2}},k::Integer) where {S1,S2<:Space{D,R}} where {D,R<:AbstractArray} =
    sp.spaces[1] ⊗ sp.spaces[2][k]


length(sp::TensorSpace{Tuple{S1,S2}}) where {S1<:Space{D,R},S2} where {D,R<:AbstractArray} =
    length(sp.spaces[1])

length(sp::TensorSpace{Tuple{S1,S2}}) where {S1,S2<:Space{D,R}} where {D,R<:AbstractArray} =
    length(sp.spaces[2])


iterate(sp::TensorSpace{Tuple{S1,S2}},k...) where {S1<:Space{D,R},S2} where {D,R<:AbstractArray} =
    iterate(components(sp),k...)

iterate(sp::TensorSpace{Tuple{S1,S2}},k...) where {S1,S2<:Space{D,R}} where {D,R<:AbstractArray} =
    iterate(components(sp),k...)


# every column is in the same space for a TensorSpace
# TODO: remove
columnspace(S::TensorSpace,_) = S.spaces[1]


struct ProductSpace{S<:Space,V<:Space,D,R} <: AbstractProductSpace{Tuple{S,V},D,R}
    spacesx::Vector{S}
    spacey::V
end

function ProductSpace(spacesx::AbstractVector, spacey)
    ProductSpace{eltype(spacesx),typeof(spacey),typeof(mapreduce(domain, ×, spacesx)),
                mapreduce(s->eltype(domain(s)),promote_type,spacesx)}(spacesx,spacey)
end

# TODO: This is a weird definition
⊗(A::AbstractVector{S},B::Space) where {S<:Space} = ProductSpace(A,B)
domain(f::ProductSpace) = domain(f.spacesx[1]) × domain(f.spacey)

factors(d::ProductSpace) = (d.spacesx, d.spacey)

nfactors(d::AbstractProductSpace) = length(d.spaces)
factors(d::AbstractProductSpace) = d.spaces
factor(d::AbstractProductSpace,k) = factors(d)[k]


isambiguous(A::TensorSpace) = isambiguous(A.spaces[1]) || isambiguous(A.spaces[2])


Base.transpose(d::TensorSpace) = TensorSpace(d.spaces[2],d.spaces[1])





## Transforms
function nDtransform_inner!(A, tempv, Rpre, Rpost, dim, plan!)
    for indpost in Rpost, indpre in Rpre
        v = view(A, indpre, :, indpost)
        tempv .= v
        v .= plan! * tempv
    end
    A
end
for (plan, plan!, Typ) in ((:plan_transform, :plan_transform!, :TransformPlan),
                           (:plan_itransform, :plan_itransform!, :ITransformPlan))

    for (f, ip) in [(plan, false), (plan!, true)]
        @eval function $f(S::TensorSpace{<:NTuple{N,Space}}, A::AbstractArray{<:Any,N}) where {N}
            spaces = S.spaces
            tempv = similar(A, size(A,1))
            sizehint!(tempv, reduce(max, size(A), init=0))
            plans = ntuple(N) do dim
                szdim = size(A,dim)
                resize!(tempv, szdim)
                ($f(spaces[dim], tempv), szdim)
            end
            $Typ(S, plans, Val{$ip})
        end
    end

    @eval begin
        function *(T::$Typ{<:Any,<:TensorSpace{<:NTuple{2,Space}},true}, M::AbstractMatrix)
            Base.require_one_based_indexing(M)
            all(dim -> T.plan[dim][2] == size(M,dim), 1:2) ||
                throw(ArgumentError("size of matrix is incompatible with transform plan"))

            tempv = similar(M, size(M,1))
            for k in axes(M,2)
                tempv .= @view M[:, k]
                M[:,k]=T.plan[1][1]*tempv
            end
            resize!(tempv, size(M,2))
            for k in axes(M,1)
                tempv .= @view M[k,:]
                M[k,:]=T.plan[2][1]*tempv
            end
            M
        end

        function *(T::$Typ{<:Any,<:TensorSpace{<:NTuple{N,Space}},true}, A::AbstractArray{<:Any,N}) where {N}
            Base.require_one_based_indexing(A)
            all(dim -> T.plan[dim][2] == size(A,dim), 1:N) ||
                throw(ArgumentError("size of array is incompatible with transform plan"))

            tempv = similar(A, size(A,1))
            sizehint!(tempv, reduce(max, size(A), init=0))
            for dim in 1:N
                Rpre = CartesianIndices(axes(A)[1:dim-1])
                Rpost = CartesianIndices(axes(A)[dim+1:end])
                resize!(tempv, size(A, dim))
                nDtransform_inner!(A, tempv, Rpre, Rpost, dim, T.plan[dim][1])
            end
            A
        end

        function *(T::$Typ{<:Any,<:TensorSpace{<:NTuple{N,Space}},false},
                A::AbstractArray{<:Any,N}) where {N}
            # TODO: we assume that the transform has the same number of coefficients
            # as the number of points in A
            # This may not always be the case, so we may need to fix this
            $Typ(T.space, T.plan, Val{true}) * copy(A)
        end

        function *(T::$Typ{TT,SS,false},v::AbstractVector) where {SS<:TensorSpace,TT}
            P = $Typ(T.space,T.plan,Val{true})
            P * copy(v)
        end
    end
end

function plan_transform(sp::TensorSpace, ::Type{T}, n::Integer) where {T}
    NM=n
    if isfinite(dimension(sp.spaces[1])) && isfinite(dimension(sp.spaces[2]))
        N,M=dimension(sp.spaces[1]),dimension(sp.spaces[2])
    elseif isfinite(dimension(sp.spaces[1]))
        N=dimension(sp.spaces[1])
        M=NM÷N
    elseif isfinite(dimension(sp.spaces[2]))
        M=dimension(sp.spaces[2])
        N=NM÷M
    else
        N=M=round(Int,sqrt(n))
    end

    TransformPlan(sp,((plan_transform(sp.spaces[1],T,N),N),
                    (plan_transform(sp.spaces[2],T,M),M)),
                Val{false})
end

function plan_transform!(sp::TensorSpace, ::Type{T}, n::Integer) where {T}
    P = plan_transform(sp, T, n)
    TransformPlan(sp, P.plan, Val{true})
end

plan_transform(sp::TensorSpace, v::AbstractVector) = plan_transform(sp,eltype(v),length(v))
plan_transform!(sp::TensorSpace, v::AbstractVector) = plan_transform!(sp,eltype(v),length(v))

function plan_itransform(sp::TensorSpace, v::AbstractVector{T}) where {T}
    N,M = size(totensor(sp, v)) # wasteful
    ITransformPlan(sp,((plan_itransform(sp.spaces[1],T,N),N),
                    (plan_itransform(sp.spaces[2],T,M),M)),
                Val{false})
end


function *(T::TransformPlan{TT,<:TensorSpace,true},v::AbstractVector) where TT # need where TT
    N,M = T.plan[1][2],T.plan[2][2]
    V=reshape(v,N,M)
    fromtensor(T.space,T*V)
end

*(T::ITransformPlan{TT,<:TensorSpace,true},v::AbstractVector) where TT  =
    vec(T*totensor(T.space,v))


## points

points(d::Union{EuclideanDomain{2},BivariateSpace},n,m) = points(d,n,m,1),points(d,n,m,2)

function points(d::BivariateSpace,n,m,k)
    ptsx=points(columnspace(d,1),n)
    ptst=points(factor(d,2),m)

    promote_type(eltype(ptsx),eltype(ptst))[fromcanonical(d,x,t)[k] for x in ptsx, t in ptst]
end




##  Fun routines

fromtensor(S::Space,M::AbstractMatrix) = fromtensor(tensorizer(S),M)
totensor(S::Space,M::AbstractVector) = totensor(tensorizer(S),M)
totensor(SS::TensorSpace{<:NTuple{d}},M::AbstractVector) where {d} =
        if d>2; totensoriterator(tensorizer(SS),M) else totensor(tensorizer(SS),M) end

function fromtensor(it::Tensorizer,M::AbstractMatrix)
    n,m=size(M)
    ret=zeros(eltype(M),blockstop(it,max(n,m)+1))
    k = 1
    for (K,J) in it
        if k > length(ret)
            break
        end
        if K ≤ n && J ≤ m
            ret[k] = M[K,J]
        end
        k += 1
    end
    ret
end


function totensor(it::Tensorizer,M::AbstractVector)
    n=length(M)
    B=block(it,n)

    #ret=zeros(eltype(M),[sum(it.blocks[i][1:min(B.n[1],length(it.blocks[i]))]) for i=1:length(it.blocks)]...)

    ret=zeros(eltype(M),sum(it.blocks[1][1:min(B.n[1],length(it.blocks[1]))]),
                        sum(it.blocks[2][1:min(B.n[1],length(it.blocks[2]))]))

    k=1
    for index in it
        if k > n
            break
        end
        ret[index...] = M[k]
        k += 1
    end
    ret
end

@inline function totensoriterator(it::TrivialTensorizer{d},M::AbstractVector) where {d}
    B=block(it,length(M))
    return it, M, B
end

for OP in (:block,:blockstart,:blockstop)
    @eval begin
        $OP(s::TensorSpace, ::PosInfinity) = ℵ₀
        $OP(s::TensorSpace, M::Block) = $OP(tensorizer(s),M)
        $OP(s::TensorSpace, M) = $OP(tensorizer(s),M)
    end
end

function points(sp::TensorSpace,n)
    pts=Array{float(eltype(domain(sp)))}(undef,0)
    a,b = sp.spaces
    if isfinite(dimension(a)) && isfinite(dimension(b))
        N,M=dimension(a),dimension(b)
    elseif isfinite(dimension(a))
        N=dimension(a)
        M=n÷N
    elseif isfinite(dimension(b))
        M=dimension(b)
        N=n÷M
    else
        N=M=round(Int,sqrt(n))
    end

    for y in points(b,M),
        x in points(a,N)
        push!(pts,SVector(x...,y...))
    end
    pts
end


itransform(sp::TensorSpace,cfs::AbstractVector) = vec(itransform!(sp,coefficientmatrix(Fun(sp,cfs))))

# 2D evaluation functions
evaluate(f::AbstractVector,S::TensorSpace2D,x) = ProductFun(totensor(S,f), S)(x...)
evaluate(f::AbstractVector,S::TensorSpace2D,x,y) = ProductFun(totensor(S,f),S)(x,y)

# ND evaluation functions of Trivial Spaces
evaluate(f::AbstractVector,S::TensorSpaceND,x) = TrivialTensorFun(totensor(S, f)..., S)(x...)

coefficientmatrix(f::Fun{<:AbstractProductSpace}) = totensor(space(f),f.coefficients)



#TODO: Implement
# function ∂(d::TensorSpace{<:IntervalOrSegment{Float64}})
#     @assert length(d.spaces) ==2
#     PiecewiseSpace([d[1].a+im*d[2],d[1].b+im*d[2],d[1]+im*d[2].a,d[1]+im*d[2].b])
# end


union_rule(a::TensorSpace,b::TensorSpace) = TensorSpace(map(union,a.spaces,b.spaces))



## Convert from 1D to 2D


# function isconvertible{T,TT}(sp::Space{Segment{SVector{2,TT}},<:Real},ts::TensorSpace)
#     d1 = domain(sp)
#     d2 = domain(ts)
#     if d2
#     length(ts.spaces) == 2 &&
#     ((domain(ts)[1] == Point(0.0) && isconvertible(sp,ts.spaces[2])) ||
#      (domain(ts)[2] == Point(0.0) && isconvertible(sp,ts.spaces[1])))
#  end

isconvertible(sp::UnivariateSpace,ts::TensorSpace{SV,D,R}) where {SV,D<:EuclideanDomain{2},R} = length(ts.spaces) == 2 &&
    ((domain(ts)[1] == Point(0.0) && isconvertible(sp,ts.spaces[2])) ||
     (domain(ts)[2] == Point(0.0) && isconvertible(sp,ts.spaces[1])))


# coefficients(f::AbstractVector,sp::ConstantSpace,ts::TensorSpace{SV,D,R}) where {SV,D<:EuclideanDomain{2},R} =
#     f[1]*ones(ts).coefficients

#
# function coefficients(f::AbstractVector,sp::Space{IntervalOrSegment{SVector{2,TT}}},ts::TensorSpace{Tuple{S,V},D,R}) where {S,V<:ConstantSpace,D<:EuclideanDomain{2},R,TT} where {T<:Number}
#     a = domain(sp)
#     b = domain(ts)
#     # make sure we are the same domain. This will be replaced by isisomorphic
#     @assert first(a) ≈ SVector(first(factor(b,1)),factor(b,2).x) &&
#         last(a) ≈ SVector(last(factor(b,1)),factor(b,2).x)
#
#     coefficients(f,sp,setdomain(factor(ts,1),a))
# end


function coefficients(f::AbstractVector,sp::UnivariateSpace,ts::TensorSpace{SV,D,R}) where {SV,D<:EuclideanDomain{2},R}
    @assert length(ts.spaces) == 2

    if factor(domain(ts),1) == Point(0.0)
        coefficients(f,sp,ts.spaces[2])
    elseif factor(domain(ts),2) == Point(0.0)
        coefficients(f,sp,ts.spaces[1])
    else
        error("Cannot convert coefficients from $sp to $ts")
    end
end


function isconvertible(sp::Space{Segment{SVector{2,TT}}},ts::TensorSpace{SV,D,R}) where {TT,SV,D<:EuclideanDomain{2},R}
    d1 = domain(sp)
    d2 = domain(ts)
    if length(ts.spaces) ≠ 2
        return false
    end
    if d1.a[2] ≈ d1.b[2]
        isa(factor(d2,2),Point) && factor(d2,2).x ≈ d1.a[2] &&
            isconvertible(setdomain(sp,Segment(d1.a[1],d1.b[1])),ts[1])
    elseif d1.a[1] ≈ d1.b[1]
        isa(factor(d2,1),Point) && factor(d2,1).x ≈ d1.a[1] &&
            isconvertible(setdomain(sp,Segment(d1.a[2],d1.b[2])),ts[2])
    else
        return false
    end
end


function coefficients(f::AbstractVector,sp::Space{Segment{SVector{2,TT}}},
                            ts::TensorSpace{SV,D,R}) where {TT,SV,D<:EuclideanDomain{2},R}
    @assert length(ts.spaces) == 2
    d1 = domain(sp)
    d2 = domain(ts)
    if d1.a[2] ≈ d1.b[2]
        coefficients(f,setdomain(sp,Segment(d1.a[1],d1.b[1])),factor(ts,1))
    elseif d1.a[1] ≈ d1.b[1]
        coefficients(f,setdomain(sp,Segment(d1.a[2],d1.b[2])),factor(ts,2))
    else
        error("Cannot convert coefficients from $sp to $ts")
    end
end




Fun(::typeof(identity), S::TensorSpace) = Fun(collect, S)
