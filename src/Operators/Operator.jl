export Operator
export bandwidths, bandrange, \, periodic
export ldirichlet,rdirichlet,lneumann,rneumann
export ldiffbc,rdiffbc
export domainspace,rangespace

const VectorIndices = Union{AbstractRange, Colon}
const IntOrVectorIndices = Union{Integer, VectorIndices}

"""
    Operator{T}

Abstract type to represent linear operators between spaces.
"""
abstract type Operator{T} end #T is the entry type, Float64 or Complex{Float64}

const VectorOrTupleOfOp{O<:Operator} = Union{AbstractVector{O}, Tuple{O, Vararg{O}}}
const ArrayOrTupleOfOp{O<:Operator} = Union{AbstractArray{O}, Tuple{O, Vararg{O}}}

eltype(::Type{<:Operator{T}}) where {T} = T

promote_eltypeof(As::ArrayOrTupleOfOp{<:Operator{T}}) where {T} = T

# default entry type
# we assume entries depend on both the domain and the basis
# realdomain case doesn't use


prectype(sp::Space) = promote_type(prectype(domaintype(sp)),eltype(rangetype(sp)))

 #Operators are struct
copy(A::Operator) = A


BroadcastStyle(::Type{<:Operator}) = DefaultArrayStyle{2}()
broadcastable(A::Operator) = A

spaces(A::Operator) = (rangespace(A), domainspace(A)) # order is consistent with size(::Matrix)
domain(A::Operator) = domain(domainspace(A))


isconstspace(_) = false
## Functionals
isafunctional(A::Operator)::Bool = size(A,1)==1 && isconstspace(rangespace(A))


isonesvec(A) = A isa AbstractFill && getindex_value(A) == 1
# block lengths of a space are 1
hastrivialblocks(A::Space) = isonesvec(blocklengths(A))
hastrivialblocks(A::Operator) = hastrivialblocks(domainspace(A)) &&
                                hastrivialblocks(rangespace(A))

# blocklengths are constant lengths
hasconstblocks(A::Space) = isa(blocklengths(A),AbstractFill)
hasconstblocks(A::Operator) = hasconstblocks(domainspace(A)) && hasconstblocks(rangespace(A)) &&
                                getindex_value(blocklengths(domainspace(A))) == getindex_value(blocklengths(rangespace(A)))


# Operator traits

abstract type OperatorStyle end

struct DefaultStyle <: OperatorStyle end

struct OperatorWrapperStyle{Structure,Indexing,Spaces} <: OperatorStyle end
const WrapperStructure = OperatorWrapperStyle{true}
const WrapperIndexing = OperatorWrapperStyle{<:Any,true}
const WrapperSpaces = OperatorWrapperStyle{<:Any,<:Any,true}
const WrapperStructureIndexing = OperatorWrapperStyle{true,true}
const WrapperStructureIndexingSpaces = OperatorWrapperStyle{true,true,true}

OperatorWrapperStyle(A::Operator) = OperatorWrapperStyle(typeof(A))
function OperatorWrapperStyle(::Type{T}) where {T<:Operator}
    Structure = iswrapperstructure(T)
    Indexing = iswrapperindexing(T)
    Spaces = iswrapperspaces(T)
    OperatorWrapperStyle{Structure,Indexing,Spaces}()
end

for f in [:iswrapperstructure, :iswrapperindexing, :iswrapperspaces]
    @eval $f(@nospecialize(_)) = false
end

iswrapperstructure(A::Operator) = iswrapperstructure(typeof(A))
iswrapperindexing(A::Operator) = iswrapperindexing(typeof(A))
iswrapperspaces(A::Operator) = iswrapperspaces(typeof(A))
iswrapper(A) = iswrapperstructure(A) || iswrapperindexing(A) || iswrapperspaces(A)

struct Functional <: OperatorStyle end

isafunctional(@nospecialize(_)) = false

# This disambiguates cases where an operator is both a functional and a wrapper
# In such cases, functions get to choose which trait to dispatch on
# All such operators must disambiguate between conflicting styles
struct StyleConflict{W, F} <: OperatorStyle
    wrapper :: W
    functional :: F
end

function dominantstyle(S::StyleConflict, f, op::Operator)
    throw(ArgumentError("please disambiguate operator style for $f and $(typeof(op))"))
end

OperatorStyle(A::Operator) = OperatorStyle(typeof(A))
function OperatorStyle(::Type{T}) where {T<:Operator}
    Wrap = OperatorWrapperStyle(T)
    f = Functional()
    if iswrapper(T) && isafunctional(T)
        StyleConflict(Wrap, f)
    elseif iswrapper(T)
        Wrap
    elseif isafunctional(T)
        f
    else
        DefaultStyle()
    end
end


## We assume operators are T->T
"""
    rangespace(op::Operator)

Return the range space of `op`.  That is, `op*f` will return a `Fun` in the
space `rangespace(op)`, provided `f` can be converted to a `Fun` in
`domainspace(op)`.
"""
rangespace(A::Operator) = oprangespace(OperatorStyle(A), A)

"""
    domainspace(op::Operator)

Return the domain space of `op`.  That is, `op*f` will first convert `f` to
a `Fun` in the space `domainspace(op)` before applying the operator.
"""
domainspace(A::Operator) = opdomainspace(OperatorStyle(A), A)

# Fallback definitions
opbandwidths(::OperatorStyle, A) = (size(A,1)-1,size(A,2)-1)

opstride(::OperatorStyle, A) = isdiag(A) ? factorial(10) : 1

opisblockbanded(::OperatorStyle, A) = all(isfinite, blockbandwidths(A))::Bool

opisbandedblockbanded(::OperatorStyle, A) =
    isbandedblockbandedabove(A) && isbandedblockbandedbelow(A)

opisbanded(::OperatorStyle, A) =
    all(isfinite, bandwidths(A))::Bool

function opisraggedbelow(::OperatorStyle, A)
    isbandedbelow(A)::Bool ||
        isbandedblockbanded(A)::Bool ||
        isblockbandedbelow(A)::Bool
end

# this should be determinable at compile time
#TODO: I think it can be generalized to the case when the domainspace
# blocklengths == rangespace blocklengths, in which case replace the definition
# of p with maximum(blocklength(domainspace(A)))
function opblockbandwidths(::OperatorStyle, A)
    hastrivialblocks(A) && return bandwidths(A)

    if hasconstblocks(A)
        a,b = bandwidths(A)
        p = getindex_value(blocklengths(domainspace(A)))
        return (-fld(-a,p),-fld(-b,p))
    end

    #TODO: Generalize to finite dimensional
    if size(A,2) == 1
        rs = rangespace(A)

        if hasconstblocks(rs)
            a = bandwidth(A,1)
            p = getindex_value(blocklengths(rs))
            return (-fld(-a,p),0)
        end
    end

    return (length(blocklengths(rangespace(A)))-1,length(blocklengths(domainspace(A)))-1)
end

opsubblockbandwidths(::OperatorStyle, A) =
    maximum(blocklengths(rangespace(A)))-1, maximum(blocklengths(domainspace(A)))-1

opbandwidth(::OperatorStyle, A, k) = bandwidths(A)[k]
opblockbandwidth(::OperatorStyle, A, k) = blockbandwidths(A)[k]
opsubblockbandwidth(::OperatorStyle, A, k) = subblockbandwidths(A)[k]

opsize(::OperatorStyle, A, k) = k==1 ? dimension(rangespace(A)) : dimension(domainspace(A))

opisdiag(::OperatorStyle, A) = bandwidths(A)==(0,0)
opissymmetric(::OperatorStyle, A) = false

for f in [:domainspace, :rangespace]
    opf = Symbol(:op, f)
    @eval begin
        $opf(::OperatorStyle, A) = error("Override $($f) for $(typeof(A))")
        $opf(S::StyleConflict, A) = $opf(dominantstyle(S, $f, A), A)
    end
end
# Since Functional doesn't change the domain space, we may forward it for WrapperSpaces
opdomainspace(S::StyleConflict{<:WrapperSpaces}, A) = opdomainspace(S.wrapper, A)

for f in [:bandwidths]
    opf = Symbol(:op, f)
    @eval begin
        # we are always banded by the size
        """
            bandwidths(op::Operator)

        Return the bandwidth of `op` in the form `(l,u)`, where `l ≥ 0` represents
        the number of subdiagonals and `u ≥ 0` represents the number of superdiagonals.
        """
        $f(A::Operator) = $opf(OperatorStyle(A), A)
        $opf(::WrapperStructure, A) = $f(A.op)
        $opf(S::StyleConflict, A) = $opf(dominantstyle(S, $f, A), A)
    end
end

## stride(::Operator)
# lets us know if operators decouple the entries
# to split into sub problems
# A diagonal operator has essentially infinite stride
# which we represent by a factorial, so that
# the gcd with any number < 10 is the number

for f in [:stride, :blockbandwidths, :subblockbandwidths,
            :israggedbelow, :isbanded, :isblockbanded, :isbandedblockbanded,
            :issymmetric, :isdiag]
    opf = Symbol(:op, f)
    @eval begin
        $f(A::Operator) = $opf(OperatorStyle(A), A)
        $opf(::WrapperStructure, A) = $f(A.op)
        $opf(S::StyleConflict, A) = $opf(dominantstyle(S, $f, A), A)
    end
end

for f in [:size, :bandwidth, :blockbandwidth, :subblockbandwidth]
    opf = Symbol(:op, f)
    @eval begin
        $f(A::Operator, k::Integer) = $opf(OperatorStyle(A), A, k)
        $opf(::WrapperStructure, A, k) = $f(A.op, k)
        $opf(S::StyleConflict, A, k) = $opf(dominantstyle(S, $f, A), A, k)
    end
end

# Functional
opsize(::Functional, A, k) = k==1 ? 1 : dimension(domainspace(A))
oprangespace(::Functional, F) = ConstantSpace(eltype(F))
opblockbandwidths(::Functional, A) = 0,hastrivialblocks(domainspace(A)) ? bandwidth(A,2) : ℵ₀

macro functional(FF)
    quote
        ApproxFunBase.isafunctional(::Type{<:$FF}) = true
        function ApproxFunBase.defaultgetindex(f::$FF,k::Integer,j::Integer)
            @assert k==1
            f[j]::eltype(f)
        end
        function ApproxFunBase.defaultgetindex(f::$FF,k::Integer,j::AbstractRange)
            @assert k==1
            f[j]
        end
        function ApproxFunBase.defaultgetindex(f::$FF,k::Integer,j)
            @assert k==1
            f[j]
        end
        function ApproxFunBase.defaultgetindex(f::$FF,k::AbstractRange,j::Integer)
            @assert k==1:1
            f[j]
        end
        function ApproxFunBase.defaultgetindex(f::$FF,k::AbstractRange,j::AbstractRange)
            @assert k==1:1
            reshape(f[j],1,length(j))
        end
        function ApproxFunBase.defaultgetindex(f::$FF,k::AbstractRange,j)
            @assert k==1:1
            reshape(f[j],1,length(j))
        end
    end
end


blocksize(A::Operator,k) = k==1 ? length(blocklengths(rangespace(A))) : length(blocklengths(domainspace(A)))
blocksize(A::Operator) = (blocksize(A,1),blocksize(A,2))

# operators need to define size(A, k::Integer)
size(A::Operator) = (size(A,1),size(A,2))
length(A::Operator) = size(A,1) * size(A,2)

# used to compute "end" for last index
function lastindex(A::Operator, n::Integer)
    if n > 2
        1
    elseif n==2
        size(A,2)
    elseif isinf(size(A,2)) || isinf(size(A,1))
        ℵ₀
    else
        size(A,1)
    end
end
lastindex(A::Operator) = size(A,1)*size(A,2)

ndims(::Operator) = 2






## bandrange and indexrange
isbandedbelow(A::Operator) = isfinite(bandwidth(A,1))::Bool
isbandedabove(A::Operator) = isfinite(bandwidth(A,2))::Bool

isbandedblockbandedbelow(_) = false
isbandedblockbandedabove(_) = false

isblockbandedbelow(A::Operator) = isfinite(blockbandwidth(A,1))::Bool
isblockbandedabove(A::Operator) = isfinite(blockbandwidth(A,2))::Bool



istriu(A::Operator) = bandwidth(A, 1) <= 0
istril(A::Operator) = bandwidth(A, 2) <= 0


## Construct operators


include("SubOperator.jl")

## getindex


"""
    (op::Operator)[k,j]

Return the `k`th coefficient of `op*Fun([zeros(j-1);1],domainspace(op))`.
"""
getindex(B::Operator,k,j) = defaultgetindex(B,k,j)
getindex(B::Operator,k) = defaultgetindex(B,k)
getindex(B::Operator,k::Block{2}) = B[Block.(k.n)...]




## override getindex.

defaultgetindex(B::Operator,k::Integer) =
    error("Override getindex(::$(typeof(B)), ::Integer)")
defaultgetindex(B::Operator,k::Integer,j::Integer) =
    error("Override getindex(::$(typeof(B)), ::Integer, ::Integer)")
defaultgetindex(A::Operator,kj::CartesianIndex) = A[Tuple(kj)...]

# Ranges

index_ndim(::Integer) = 0
index_ndim(::Union{AbstractVector, Block, Colon}) = 1
index_ndims(inds...) = Val(sum(index_ndim, inds))

select_vectorinds(a, b...) = a
select_vectorinds(a::Integer, b...) = select_vectorinds(b...)

replace_vector_by_scalar(inds::Tuple, k) = (k, replace_vector_by_scalar(inds[2:end], k)...)
replace_vector_by_scalar(inds::Tuple{Integer,Vararg}, k) = (inds[1], replace_vector_by_scalar(inds[2:end], k)...)
replace_vector_by_scalar(::Tuple{}, k) = ()

defaultgetindex(B::Operator, kj...) = defaultgetindex(B, index_ndims(kj...), kj...)

function defaultgetindex(op::Operator, ::Val{1}, inds...)
    indsvec = select_vectorinds(inds...)
    # avoid stack-overflow if iterating over indsvec returns indsvec
    # e.g. Blocks
    k = first(indsvec)
    indsscal = replace_vector_by_scalar(inds, k)
    if typeof(indsscal) == typeof(inds)
        T = typeof(op)
        throw(ArgumentError("please implement "*
            "getindex(::$T, $(join(string.("::", typeof.(inds)), ",")))"))
    end
    eltype(op)[op[replace_vector_by_scalar(inds, k)...] for k in indsvec]
end

function defaultgetindex(B::Operator, ::Val{2}, inds...)
    S = view(B,inds...)
    all(isfinite, size(S)) || return S
    AbstractMatrix(S)
end

# TODO: finite dimensional blocks
blockcolstart(A::Operator, J::Block{1}) = Block(max(1,Int(J)-blockbandwidth(A,2)))
blockrowstart(A::Operator, K::Block{1}) = Block(max(1,Int(K)-blockbandwidth(A,1)))
blockcolstop(A::Operator, J::Block{1}) = Block(min(Int(J)+blockbandwidth(A,1),blocksize(A,1)))
blockrowstop(A::Operator, K::Block{1}) = Block(min(Int(K)+blockbandwidth(A,2),blocksize(A,2)))

blockrows(A::Operator, K::Block{1}) = blockrange(rangespace(A),K)
blockcols(A::Operator, J::Block{1}) = blockrange(domainspace(A),J)


# default is to use bandwidth
# override for other shaped operators
#TODO: Why size(A,2) in colstart?
banded_colstart(A::Operator, i::Integer) = min(max(i-bandwidth(A,2), 1), size(A, 2))
banded_colstop(A::Operator, i::Integer) = max(0,min(i+bandwidth(A,1), size(A, 1)))
banded_rowstart(A::Operator, i::Integer) = min(max(i-bandwidth(A,1), 1), size(A, 1))
banded_rowstop(A::Operator, i::Integer) = max(0,min(i+bandwidth(A,2), size(A, 2)))

blockbanded_colstart(A::Operator, i::Integer) =
        blockstart(rangespace(A), block(domainspace(A),i)-blockbandwidth(A,2))
blockbanded_colstop(A::Operator, i::Integer) =
    min(blockstop(rangespace(A), block(domainspace(A),i)+blockbandwidth(A,1)),
        size(A, 1))
blockbanded_rowstart(A::Operator, i::Integer) =
        blockstart(domainspace(A), block(rangespace(A),i)-blockbandwidth(A,1))
blockbanded_rowstop(A::Operator, i::Integer) =
    min(blockstop(domainspace(A), block(rangespace(A),i)+blockbandwidth(A,2)),
        size(A, 2))


function bandedblockbanded_colstart(A::Operator, i::Integer)
    ds = domainspace(A)
    B = block(ds,i)
    ξ = i - blockstart(ds,B) + 1  # col in block
    bs = blockstart(rangespace(A), B-blockbandwidth(A,2))
    max(bs,bs + ξ - 1 - subblockbandwidth(A,2))
end

function bandedblockbanded_colstop(A::Operator, i::Integer)
    i ≤ 0 && return 0
    ds = domainspace(A)
    rs = rangespace(A)
    B = block(ds,i)
    ξ = i - blockstart(ds,B) + 1  # col in block
    Bend = B+blockbandwidth(A,1)
    bs = blockstart(rs, Bend)
    min(blockstop(rs,Bend),bs + ξ - 1 + subblockbandwidth(A,1))
end

function bandedblockbanded_rowstart(A::Operator, i::Integer)
    rs = rangespace(A)
    B = block(rs,i)
    ξ = i - blockstart(rs,B) + 1  # row in block
    bs = blockstart(domainspace(A), B-blockbandwidth(A,1))
    max(bs,bs + ξ - 1 - subblockbandwidth(A,1))
end

function bandedblockbanded_rowstop(A::Operator, i::Integer)
    ds = domainspace(A)
    rs = rangespace(A)
    B = block(rs,i)
    ξ = i - blockstart(rs,B) + 1  # row in block
    Bend = B+blockbandwidth(A,2)
    bs = blockstart(ds, Bend)
    min(blockstop(ds,Bend),bs + ξ - 1 + subblockbandwidth(A,2))
end


unstructured_colstart(A, i) = 1
unstructured_colstop(A, i) = size(A,1)
unstructured_rowstart(A, i) = 1
unstructured_rowstop(A, i) = size(A,2)


function default_colstart(A::Operator, i::Integer)
    if isbandedabove(A)
        banded_colstart(A,i)
    elseif isbandedblockbanded(A)
        bandedblockbanded_colstart(A, i)
    elseif isblockbanded(A)
        blockbanded_colstart(A, i)
    else
        unstructured_colstart(A, i)
    end
end

function default_colstop(A::Operator, i::Integer)
    if isbandedbelow(A)
        banded_colstop(A,i)
    elseif isbandedblockbanded(A)
        bandedblockbanded_colstop(A, i)
    elseif isblockbanded(A)
        blockbanded_colstop(A, i)
    else
        unstructured_colstop(A, i)
    end
end

function default_rowstart(A::Operator, i::Integer)
    if isbandedbelow(A)
        banded_rowstart(A,i)
    elseif isbandedblockbanded(A)
        bandedblockbanded_rowstart(A, i)
    elseif isblockbanded(A)
        blockbanded_rowstart(A, i)
    else
        unstructured_rowstart(A, i)
    end
end

function default_rowstop(A::Operator, i::Integer)
    if isbandedabove(A)
        banded_rowstop(A,i)
    elseif isbandedblockbanded(A)
        bandedblockbanded_rowstop(A, i)
    elseif isblockbanded(A)
        blockbanded_rowstop(A, i)
    else
        unstructured_rowstop(A, i)
    end
end



for OP in (:colstart,:colstop,:rowstart,:rowstop)
    defOP = Symbol(:default_, OP)
    @eval begin
        $OP(A::Operator, i::Integer) = $defOP(A,i)
        $OP(A::Operator, i::PosInfinity) = ℵ₀
    end
end




function defaultgetindex(A::Operator,::Type{FiniteRange},::Type{FiniteRange})
    if isfinite(size(A,1)) && isfinite(size(A,2))
        A[1:size(A,1),1:size(A,2)]
    else
        error("Only exists for finite operators.")
    end
end

defaultgetindex(A::Operator,k::Type{FiniteRange},J::Block) = A[k,blockcols(A,J)]
function defaultgetindex(A::Operator,::Type{FiniteRange},jr::AbstractVector{Int})
    cs = (isbanded(A) || isblockbandedbelow(A)) ? colstop(A,maximum(jr)) : mapreduce(j->colstop(A,j),max,jr)
    A[1:cs,jr]
end

function defaultgetindex(A::Operator,::Type{FiniteRange},jr::BlockRange{1})
    cs = (isbanded(A) || isblockbandedbelow(A)) ? blockcolstop(A,maximum(jr)) : mapreduce(j->blockcolstop(A,j),max,jr)
    A[Block(1):cs,jr]
end

function view(A::Operator,::Type{FiniteRange},jr::AbstractVector{Int})
    cs = (isbanded(A) || isblockbandedbelow(A)) ? colstop(A,maximum(jr)) : mapreduce(j->colstop(A,j),max,jr)
    view(A,1:cs,jr)
end

function view(A::Operator,::Type{FiniteRange},jr::BlockRange{1})
    cs = (isbanded(A) || isblockbandedbelow(A)) ? blockcolstop(A,maximum(jr)) : mapreduce(j->blockcolstop(A,j),max,jr)
    view(A,Block(1):cs,jr)
end


defaultgetindex(A::Operator,K::Block,j::Type{FiniteRange}) = A[blockrows(A,K),j]
defaultgetindex(A::Operator,kr,::Type{FiniteRange}) =
    A[kr,1:rowstop(A,maximum(kr))]





## Composition with a Fun, LowRankFun, and ProductFun
"""
    (op::Operator)[f::Fun]

Construct the operator `op * Multiplication(f)`, that is, it multiplies on the right
by `f` first.  Note that `op * f` is different: it applies `op` to `f`.

# Examples
```jldoctest
julia> x = Fun()
Fun(Chebyshev(), [0.0, 1.0])

julia> D = Derivative()
ConcreteDerivative : ApproxFunBase.UnsetSpace() → ApproxFunBase.UnsetSpace()

julia> Dx = D[x] # construct the operator y -> d/dx * (x * y)
TimesOperator : ApproxFunBase.UnsetSpace() → ApproxFunBase.UnsetSpace()

julia> twox = Dx * x # Evaluate d/dx * (x * x)
Fun(Ultraspherical(1), [0.0, 1.0])

julia> twox(0.1) ≈ 2 * 0.1
true
```
"""
getindex(B::Operator,f::Fun) = B*Multiplication(domainspace(B),f)
getindex(B::Operator,f::LowRankFun) = mapreduce(((fAi,fBi),) -> fAi * B[fBi], +, zip(f.A, f.B))
function getindex(B::Operator{BT}, f::ProductFun{S,V,SS,T}) where {BT,S,V,SS,T}
    TBF = promote_type(BT,T)
    sp2 = factors(f.space)[2]
    mapreduce(((ind, fi),)-> fi * B[Fun(sp2, [zeros(TBF,i-1); one(TBF)])], +,
                enumerate(f.coefficients))
end


# Convenience for wrapper ops
unwrap_axpy!(α,P,A) = axpy!(α,view(parent(P).op,P.indexes[1],P.indexes[2]),A)

# use this for wrapper operators that have the same structure but
# not necessarily the same entries
#
#  Ex: c*op or real(op)
macro wrapperstructure(Wrap)
    fns2 = [:(ApproxFunBase.colstart),:(ApproxFunBase.colstop),
             :(ApproxFunBase.rowstart),:(ApproxFunBase.rowstop)]

    v = map(fns2) do func
        quote
             $func(D::$Wrap,k::Integer) = $func(D.op,k)
             $func(A::$Wrap,i::ApproxFunBase.PosInfinity) = ℵ₀ # $func(A.op,i) | see PR #42
        end
     end

    ret = quote
        ApproxFunBase.iswrapperstructure(::Type{<:$Wrap}) = true
        $(v...)
    end

    esc(ret)
end



# use this for wrapper operators that have the same entries but
# not necessarily the same spaces
#
macro wrappergetindex(Wrap)
    v = map((:(ApproxFunBase.BandedMatrix),:(ApproxFunBase.RaggedMatrix),
                :(Base.Matrix),:(Base.Vector),:(Base.AbstractVector))) do TYP
        quote
            $TYP(P::ApproxFunBase.SubOperator{T,OP}) where {T,OP<:$Wrap} =
                $TYP(view(parent(P).op,P.indexes[1],P.indexes[2]))
            $TYP(P::ApproxFunBase.SubOperator{T,OP,NTuple{2,UnitRange{Int}}}) where {T,OP<:$Wrap} =
                $TYP(view(parent(P).op,P.indexes[1],P.indexes[2]))
        end
    end

    ret = quote
        $(v...)

        Base.getindex(OP::$Wrap,k::Integer...) =
            OP.op[k...]::eltype(OP)

        Base.getindex(OP::$Wrap,k::Union{Number,AbstractArray,Colon}...) = OP.op[k...]
        Base.getindex(OP::$Wrap,k::ApproxFunBase.InfRanges, j::ApproxFunBase.InfRanges) = view(OP, k, j)
        Base.getindex(OP::$Wrap,k::ApproxFunBase.InfRanges, j::Colon) = view(OP, k, j)
        Base.getindex(OP::$Wrap,k::Colon, j::ApproxFunBase.InfRanges) = view(OP, k, j)
        Base.getindex(OP::$Wrap,k::Colon, j::Colon) = view(OP, k, j)

        LinearAlgebra.axpy!(α,P::ApproxFunBase.SubOperator{T,OP},A::AbstractMatrix) where {T,OP<:$Wrap} =
            ApproxFunBase.unwrap_axpy!(α,P,A)

        ApproxFunBase.mul_coefficients(A::$Wrap,b) = ApproxFunBase.mul_coefficients(A.op,b)
        ApproxFunBase.mul_coefficients(A::ApproxFunBase.SubOperator{T,OP,NTuple{2,UnitRange{Int}}},b) where {T,OP<:$Wrap} =
            ApproxFunBase.mul_coefficients(view(parent(A).op,A.indexes[1],A.indexes[2]),b)
        ApproxFunBase.mul_coefficients(A::ApproxFunBase.SubOperator{T,OP},b) where {T,OP<:$Wrap} =
            ApproxFunBase.mul_coefficients(view(parent(A).op,A.indexes[1],A.indexes[2]),b)

        # fast converts to banded matrices would be based on indices, not blocks
        function ApproxFunBase.BandedMatrix(S::ApproxFunBase.SubOperator{T,OP,NTuple{2,ApproxFunBase.BlockRange1}}) where {T,OP<:$Wrap}
            A = parent(S)
            ds = ApproxFunBase.domainspace(A)
            rs = ApproxFunBase.rangespace(A)
            KR,JR = parentindices(S)
            ApproxFunBase.BandedMatrix(view(A,
                              ApproxFunBase.blockstart(rs,first(KR)):ApproxFunBase.blockstop(rs,last(KR)),
                              ApproxFunBase.blockstart(ds,first(JR)):ApproxFunBase.blockstop(ds,last(JR))))
        end


        # if the spaces change, then we need to be smarter
        function ApproxFunBase.BlockBandedMatrix(S::ApproxFunBase.SubOperator{T,OP}) where {T,OP<:$Wrap}
            ApproxFunBase._blockmaybebandedmatrix(S,
                ApproxFunBase.BlockBandedMatrix,
                ApproxFunBase.default_BlockBandedMatrix)
        end

        function ApproxFunBase.PseudoBlockMatrix(S::ApproxFunBase.SubOperator{T,OP}) where {T,OP<:$Wrap}
            ApproxFunBase._blockmaybebandedmatrix(S,
                ApproxFunBase.PseudoBlockMatrix,
                ApproxFunBase.default_BlockMatrix)
        end

        function ApproxFunBase.BandedBlockBandedMatrix(S::ApproxFunBase.SubOperator{T,OP}) where {T,OP<:$Wrap}
            ApproxFunBase._blockmaybebandedmatrix(S,
                ApproxFunBase.BandedBlockBandedMatrix,
                ApproxFunBase.default_BandedBlockBandedMatrix)
        end

        ApproxFunBase.@wrapperstructure($Wrap) # structure is automatically inherited

        ApproxFunBase.iswrapperindexing(::Type{<:$Wrap}) = true
    end

    esc(ret)
end

function _blockmaybebandedmatrix(S, f::T, fdef::D) where {T,D}
    P = parent(S)
    if ApproxFunBase.blocklengths(domainspace(P)) === ApproxFunBase.blocklengths(domainspace(P.op)) &&
            ApproxFunBase.blocklengths(rangespace(P)) === ApproxFunBase.blocklengths(rangespace(P.op))
        f(view(parent(S).op,S.indexes[1],S.indexes[2]))
    else
        fdef(S)
    end
end

isconstop(A) = opisconstop(OperatorStyle(A), A)
opisconstop(::OperatorStyle, A) = false

# use this for wrapper operators that have the same spaces but
# not necessarily the same entries or structure
#
for f in [:domainspace, :rangespace, :opisconstop]
    opf = Symbol(:op, f)
    @eval $opf(::WrapperSpaces, A) = $f(A.op)
end

macro wrapperspaces(Wrap, forwarddomain = true, forwardrange = true)
    ret = quote
        ApproxFunBase.iswrapperspaces(::Type{<:$Wrap}) = true
        # ApproxFunBase.domain(D::$Wrap) = domain(domainspace(D))
    end

    esc(ret)
end


# use this for wrapper operators that have the same entries and same spaces
#
macro wrapper(Wrap, forwarddomain = true, forwardrange = true)
    ret = quote
        ApproxFunBase.@wrappergetindex($Wrap)
        ApproxFunBase.@wrapperspaces($Wrap, $forwarddomain, $forwardrange)
    end


    esc(ret)
end

unwrap(A::Operator) = iswrapper(A) ? A.op : A

## Standard Operators and linear algebra



include("ldiv.jl")

include("spacepromotion.jl")
include("banded/banded.jl")
include("general/general.jl")

include("functionals/functionals.jl")
include("almostbanded/almostbanded.jl")

include("systems.jl")

include("qr.jl")
include("nullspace.jl")




## Conversion



zero(::Type{Operator{T}}) where {T<:Number} = ZeroOperator(T)
zero(::Type{O}) where {O<:Operator} = ZeroOperator(eltype(O))


Operator(L::UniformScaling) = ConstantOperator(L, UnsetSpace())
Operator(L::UniformScaling, s::Space) = ConstantOperator(L, s)
Operator(L::UniformScaling{Bool}, s::Space) = L.λ ? IdentityOperator(s) : ZeroOperator(s)
Operator(L::UniformScaling, d::Domain) = Operator(L, Space(d))

Operator{T}(f::Fun) where {T} =
    norm(f.coefficients)==0 ? zero(Operator{T}) : strictconvert(Operator{T}, Multiplication(f))

Operator(f::Fun) = norm(f.coefficients)==0 ? ZeroOperator() : Multiplication(f)

convert(::Type{O}, f::Fun) where O<:Operator = O(f)
Operator{T}(A::Operator) where T = strictconvert(Operator{T}, A)


## Promotion





promote_rule(::Type{N},::Type{Operator}) where {N<:Number} = Operator{N}
promote_rule(::Type{UniformScaling{N}},::Type{Operator}) where {N<:Number} =
    Operator{N}
promote_rule(::Type{Fun{S,N,VN}},::Type{Operator}) where {S,N<:Number,VN} = Operator{N}
promote_rule(::Type{N},::Type{O}) where {N<:Number,O<:Operator} =
    Operator{promote_type(N,eltype(O))}  # float because numbers are promoted to Fun
promote_rule(::Type{UniformScaling{N}},::Type{O}) where {N<:Number,O<:Operator} =
    Operator{promote_type(N,eltype(O))}
promote_rule(::Type{Fun{S,N,VN}},::Type{O}) where {S,N<:Number,O<:Operator,VN} =
    Operator{promote_type(N,eltype(O))}

promote_rule(::Type{BO1},::Type{BO2}) where {BO1<:Operator,BO2<:Operator} =
    Operator{promote_type(eltype(BO1),eltype(BO2))}




## Wrapper

#TODO: Should cases that modify be included?
const WrapperOperator = Union{SpaceOperator,MultiplicationWrapper,DerivativeWrapper,IntegralWrapper,
                                    ConversionWrapper,ConstantTimesOperator,TransposeOperator}





# The following support converting an Operator to a Matrix or BandedMatrix

## BLAS and matrix routines
# We assume that copy may be overriden

function axpy!(a, X::Operator, Y::AbstractMatrix)
    Y .+= a .* AbstractMatrix(X)
    # the explicit return statement improves type inference
    return Y
end
copyto!(dest::AbstractMatrix, src::Operator) = copyto!(dest, AbstractMatrix(src))

# this is for operators that implement copy via axpy!

function BandedMatrix(::Type{Zeros}, V::Operator)
    all(isfinite, size(V)) || throw(ArgumentError("operator must be finite"))
    BandedMatrix(Zeros{eltype(V)}(size(V)), bandwidths(V))
end
function Matrix(::Type{Zeros}, V::Operator)
    all(isfinite, size(V)) || throw(ArgumentError("operator must be finite"))
    Matrix(Zeros{eltype(V)}(size(V)))
end
function BandedBlockBandedMatrix(::Type{Zeros}, V::Operator)
    all(isfinite, size(V)) || throw(ArgumentError("operator must be finite"))
    BandedBlockBandedMatrix(Zeros{eltype(V)}(size(V)),
                            blocklengths(rangespace(V)), blocklengths(domainspace(V)),
                            blockbandwidths(V), subblockbandwidths(V))
end
function BlockBandedMatrix(::Type{Zeros}, V::Operator)
    all(isfinite, size(V)) || throw(ArgumentError("operator must be finite"))
    BlockBandedMatrix(Zeros{eltype(V)}(size(V)),
                      AbstractVector{Int}(blocklengths(rangespace(V))),
                       AbstractVector{Int}(blocklengths(domainspace(V))),
                      blockbandwidths(V))
end
function RaggedMatrix(::Type{Zeros}, V::Operator)
    all(isfinite, size(V)) || throw(ArgumentError("operator must be finite"))
    RaggedMatrix(Zeros{eltype(V)}(size(V)),
                 Int[max(0,colstop(V,j)) for j=1:size(V,2)])
end


convert_axpy!(::Type{MT}, S::Operator) where {MT <: AbstractMatrix} =
        axpy!(one(eltype(S)), S, MT(Zeros, S))



BandedMatrix(S::Operator) = default_BandedMatrix(S)

function BlockBandedMatrix(S::Operator)
    if isbandedblockbanded(S)
        BlockBandedMatrix(BandedBlockBandedMatrix(S))
    else
        default_BlockBandedMatrix(S)
    end
end

function default_BlockMatrix(S::Operator)
    ret = PseudoBlockArray(zeros(eltype(S), size(S)),
                        AbstractVector{Int}(blocklengths(rangespace(S))),
                        AbstractVector{Int}(blocklengths(domainspace(S))))
    ret .= S
    ret
end

function PseudoBlockMatrix(S::Operator)
    if isbandedblockbanded(S)
        PseudoBlockMatrix(BandedBlockBandedMatrix(S))
    elseif isblockbanded(S)
        PseudoBlockMatrix(BlockBandedMatrix(S))
    else
        default_BlockMatrix(S)
    end
end


# TODO: Unify with SubOperator
for TYP in (:RaggedMatrix, :Matrix)
    def_TYP = Symbol(string("default_", TYP))
    @eval function $TYP(S::Operator)
        if isinf(size(S,1)) || isinf(size(S,2))
            error("Cannot convert $S to a ", $TYP)
        end

        if isbanded(S)
            $TYP(BandedMatrix(S))
        else
            $def_TYP(S)
        end
    end
end

function Vector(S::Operator)
    if size(S,2) ≠ 1  || isinf(size(S,1))
        error("Cannot convert $S to a AbstractVector")
    end

    eltype(S)[S[k] for k=1:size(S,1)]
end

convert(::Type{AA}, B::Operator) where AA<:AbstractArray = AA(B)


# TODO: template out fully
arraytype(::Operator) = Matrix
function arraytype(V::SubOperator{T,B,Tuple{KR,JR}}) where {T, B, KR <: Union{BlockRange, Block}, JR <: Union{BlockRange, Block}}
    P = parent(V)
    isbandedblockbanded(P) && return BandedBlockBandedMatrix
    isblockbanded(P) && return BlockBandedMatrix
    return PseudoBlockMatrix
end

function arraytype(V::SubOperator{T,B,Tuple{KR,JR}}) where {T, B, KR <: Block, JR <: Block}
    P = parent(V)
    isbandedblockbanded(V) && return BandedMatrix
    return Matrix
end


function arraytype(V::SubOperator)
    P = parent(V)
    isbanded(P) && return BandedMatrix
    # isbandedblockbanded(P) && return BandedBlockBandedMatrix
    isinf(size(P,1)) && israggedbelow(P) && return RaggedMatrix
    return Matrix
end

AbstractMatrix(V::Operator) = arraytype(V)(V)
AbstractVector(S::Operator) = Vector(S)




# default copy is to loop through
# override this for most operators.
function default_BandedMatrix(S::Operator)
    Y=BandedMatrix{eltype(S)}(undef, size(S), bandwidths(S))

    for j=1:size(S,2),k=colrange(Y,j)
        @inbounds inbands_setindex!(Y,S[k,j],k,j)
    end

    Y
end


# default copy is to loop through
# override this for most operators.
function default_RaggedMatrix(S::Operator)
    data=Array{eltype(S)}(undef, 0)
    cols=Array{Int}(undef, size(S,2)+1)
    cols[1]=1
    for j=1:size(S,2)
        cs=colstop(S,j)
        K=cols[j]-1
        cols[j+1]=cs+cols[j]
        resize!(data,cols[j+1]-1)

        for k=1:cs
            data[K+k]=S[k,j]
        end
    end

    RaggedMatrix(data,cols,size(S,1))
end

function default_Matrix(S::Operator)
    n, m = size(S)
    if isinf(n) || isinf(m)
        error("Cannot convert $S to a Matrix")
    end

    eltype(S)[S[k,j] for k=1:n, j=1:m]
end




# The diagonal of the operator may not be the diagonal of the sub
# banded matrix, so the following calculates the row of the
# Banded matrix corresponding to the diagonal of the original operator


diagindshift(S,kr,jr) = first(kr)-first(jr)
diagindshift(S::SubOperator) = diagindshift(S,parentindices(S)[1],parentindices(S)[2])


#TODO: Remove
diagindrow(S,kr,jr) = bandwidth(S,2)+first(jr)-first(kr)+1
diagindrow(S::SubOperator) = diagindrow(S,parentindices(S)[1],parentindices(S)[2])

# Conversion between operator types
convert(::Type{O}, c::Operator) where {O<:Operator} = c isa O ? c : O(c)::O
