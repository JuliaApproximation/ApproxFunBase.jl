export KroneckerOperator



##########
# KroneckerOperator gives the kronecker product of two 1D operators
#########

struct KroneckerOperator{TT, DS,RS,DI,RI,T} <: Operator{T}
    ops::TT
    domainspace::DS
    rangespace::RS
    domaintensorizer::DI
    rangetensorizer::RI
end


KroneckerOperator(A,B,ds::Space,rs::Space,di,ri) =
    KroneckerOperator{typeof((A, B)),typeof(ds),typeof(rs),typeof(di),typeof(ri),
                        promote_type(eltype(A),eltype(B))}((A,B),ds,rs,di,ri)
KroneckerOperator(A::Tuple,ds::Space,rs::Space,di,ri) =
    KroneckerOperator{typeof(A), typeof(ds),typeof(rs),typeof(di),typeof(ri),
                        promote_type(eltype(A[1]),eltype(A[2]))}(A,ds,rs,di,ri)

KroneckerOperator(A,B,ds::Space,rs::Space) = KroneckerOperator(A,B,ds,rs,
                    CachedIterator(tensorizer(ds)),CachedIterator(tensorizer(rs)))
KroneckerOperator(A::Tuple, ds::Space, rs::Space) = KroneckerOperator(A,ds,rs,
                    CachedIterator(tensorizer(ds)),CachedIterator(tensorizer(rs)))

# DON'T nest KroneckerOperators
function KroneckerOperator(A::KroneckerOperator,B::KroneckerOperator)
    ds=domainspace(A)⊗domainspace(B)
    rs=rangespace(A)⊗rangespace(B)
    KroneckerOperator((A.ops..., B.ops...), ds, rs)
end
function KroneckerOperator(A::KroneckerOperator,B)
    ds=domainspace(A)⊗domainspace(B)
    rs=rangespace(A)⊗rangespace(B)
    KroneckerOperator((A.ops..., B), ds, rs)
end
function KroneckerOperator(A,B::KroneckerOperator)
    ds=domainspace(A)⊗domainspace(B)
    rs=rangespace(A)⊗rangespace(B)
    KroneckerOperator((A, B.ops...), ds, rs)
end
function KroneckerOperator(A,B)
    ds=domainspace(A)⊗domainspace(B)
    rs=rangespace(A)⊗rangespace(B)
    KroneckerOperator(A,B,ds,rs)
end
KroneckerOperator(A::UniformScaling,B::UniformScaling) =
    KroneckerOperator(ConstantOperator(A.λ),ConstantOperator(B.λ))
KroneckerOperator(A,B::UniformScaling) = KroneckerOperator(A,ConstantOperator(B.λ))
KroneckerOperator(A::UniformScaling,B) = KroneckerOperator(ConstantOperator(A.λ),B)
KroneckerOperator(A::Fun,B::Fun) = KroneckerOperator(Multiplication(A),Multiplication(B))
KroneckerOperator(A::UniformScaling,B::Fun) = KroneckerOperator(ConstantOperator(A.λ),Multiplication(B))
KroneckerOperator(A::Fun,B::UniformScaling) = KroneckerOperator(Multiplication(A),ConstantOperator(B.λ))
KroneckerOperator(A,B::Fun) = KroneckerOperator(A,Multiplication(B))
KroneckerOperator(A::Fun,B) = KroneckerOperator(Multiplication(A),B)

KroneckerOperator(K::KroneckerOperator) = K
KroneckerOperator(T::TimesOperator) = mapfoldr(op -> KroneckerOperator(op), *, T.ops)

function promotedomainspace(K::KroneckerOperator,ds::TensorSpace)
    A = (i->promoterangespace(K.ops[i], ds.spaces[i]) for i=1:length(K.ops))
    KroneckerOperator(A,ds,rangespace(K)) ## TODO: maybe rangespace(K) has to be replaces by rangespace(A[1])⊗...
end

function promoterangespace(K::KroneckerOperator,rs::TensorSpace)
    A = (i->promoterangespace(K.ops[i], rs.spaces[i]) for i=1:length(K.ops))
    KroneckerOperator(A,domainspace(K),rs)
end


function convert(::Type{Operator{T}},K::KroneckerOperator) where T<:Number
    if T == eltype(K)
        K
    else
        ops = map(Operator{T}, K.ops)
        KroneckerOperator{typeof(ops),
            typeof(K.domainspace),typeof(K.rangespace),
            typeof(K.domaintensorizer),typeof(K.rangetensorizer),T}(ops,
              K.domainspace,K.rangespace,
              K.domaintensorizer,K.rangetensorizer)::Operator{T}
    end
end


function colstart(A::KroneckerOperator,k::Integer)
    K=block(A.domaintensorizer,k)
    M = blockbandwidth(A,2)
    if isfinite(M)
        blockstart(A.rangetensorizer,max(Block(1),K-M))
    else
        blockstart(A.rangetensorizer,Block(1))
    end
end

function colstop(A::KroneckerOperator,k::Integer)
    k == 0 && return 0
    K=block(A.domaintensorizer,k)
    st=blockstop(A.rangetensorizer,blockcolstop(A,K))
    # zero indicates above dimension
    min(size(A,1),st)
end

function rowstart(A::KroneckerOperator,k::Integer)
    K=block(rangespace(A),k)
    K2 = Int(K)-blockbandwidth(A,1)
    K2 ≤ 1 && return 1
    ds = domainspace(A)
    K2 ≥ blocksize(ds,1) && return size(A,2)
    blockstart(ds,K2)
end

function rowstop(A::KroneckerOperator,k::Integer)
    K=block(rangespace(A),k)
    ds = domainspace(A)
    K2 = Int(K)+blockbandwidth(A,2)
    K2 ≥ blocksize(ds)[1] && return size(A,2)
    st=blockstop(ds,K2)
    # zero indicates above dimension
    st==0 ? size(A,2) : min(size(A,2),st)
end


bandwidths(K::KroneckerOperator) = isdiag(K) ? (0,0) : (ℵ₀,ℵ₀)

for f in [:isblockbanded, :israggedbelow, :isdiag]
    _f = Symbol(:_, f)
    @eval begin
        $f(K::KroneckerOperator) = $(_f)(K.ops)
        function $(_f)(ops::Tuple)
            $f(first(ops)) && $(_f)(Base.tail(ops))
        end
        $(_f)(::Tuple{}) = true
    end
end
isbandedblockbanded(K::KroneckerOperator) = length(K.ops)>2 ? false : _isbandedblockbanded(K.ops)
isbandedblockbandedcheck(op) = isbanded(op) && isinf(size(op,1)) && isinf(size(op,2))
function _isbandedblockbanded(ops::Tuple)
    isbandedblockbandedcheck(first(ops)) && _isbandedblockbanded(Base.tail(ops))
end
_isbandedblockbanded(::Tuple{}) = true

blockbandwidths(K::KroneckerOperator) =
    (mapreduce(k->blockbandwidth(k,1), +, K.ops), 
    mapreduce(k->blockbandwidth(k,2), +, K.ops))

# If each block were in turn BlockBandedMatrix, these would
# be the    bandwidths
# TODO: How does this work for multiple Ops?
subblock_blockbandwidths(K::KroneckerOperator) =
    (max(blockbandwidth(K.ops[1],1),blockbandwidth(K.ops[2],2)) ,
           max(blockbandwidth(K.ops[1],2),blockbandwidth(K.ops[2],1)))


# If each block were in turn BandedMatrix, these are the bandwidths
function subblockbandwidths(K::KroneckerOperator)
    isbandedblockbanded(K) || return (ℵ₀,ℵ₀)

    if all(hastrivialblocks,domainspace(K).spaces) &&
            all(hastrivialblocks,rangespace(K).spaces)
        subblock_blockbandwidths(K)
    else
        dt = domaintensorizer(K).iterator
        rt = rangetensorizer(K).iterator
        # assume block size is repeated and square
        @assert all(b->isa(b,AbstractFill),dt.blocks)
        @assert rt.blocks ≡ dt.blocks

        sb = subblock_blockbandwidths(K)
        # divide by the size of each block
        sb_sz = mapreduce(getindex_value,*,dt.blocks)
        # spread by sub block szie
        (sb[1]+1)*sb_sz-1,(sb[2]+1)*sb_sz-1
    end
end

const WrapperList = (ConversionWrapper,MultiplicationWrapper,DerivativeWrapper,LaplacianWrapper,
                       SpaceOperator,ConstantTimesOperator)
const Wrappers = Union{WrapperList...}

domaintensorizer(R::Operator) = tensorizer(domainspace(R))
rangetensorizer(R::Operator) = tensorizer(rangespace(R))

domaintensorizer(P::PlusOperator) = domaintensorizer(P.ops[1])
rangetensorizer(P::PlusOperator) = rangetensorizer(P.ops[1])

domaintensorizer(P::TimesOperator) = domaintensorizer(P.ops[end])
rangetensorizer(P::TimesOperator) = rangetensorizer(P.ops[1])

for FUNC in (:blockbandwidths, :subblockbandwidths, :isbandedblockbanded,:domaintensorizer,:rangetensorizer)
    @eval $FUNC(K::Wrappers) = $FUNC(K.op)
end

KroneckerOperator(A::Wrappers) = KroneckerOperator(A.op)

domainspace(K::KroneckerOperator) = K.domainspace
rangespace(K::KroneckerOperator) = K.rangespace

domaintensorizer(K::KroneckerOperator) = K.domaintensorizer
rangetensorizer(K::KroneckerOperator) = K.rangetensorizer


# For 2 ops, we can do this
# we suport 4-indexing with KroneckerOperator
# If A is K x J and B is N x M, then w
# index to match KO=reshape(kron(B,A),N,K,M,J)
# that is
# KO[k,n,j,m] = A[k,j]*B[n,m]

# TODO: arbitrary number of ops
# We get tuples of arbitraty length from the tensorizers

## this should not be used anymore, can be deleted
# getindex(KO::KroneckerOperator,k::Integer,n::Integer,j::Integer,m::Integer) =
#     KO.ops[1][k,j]*KO.ops[2][n,m]

function getindex(KO::KroneckerOperator,kin::Integer,jin::Integer)
    domain_tuple=KO.domaintensorizer[jin]
    range_tuple=KO.rangetensorizer[kin]
    mapreduce((k,i_in,j_out)->k[i_in,j_out], *, KO.ops, range_tuple, domain_tuple)
end

function getindex(KO::KroneckerOperator,k::Integer)
    if size(KO,1) == 1
        KO[1,k]
    elseif size(KO,2) == 1
        KO[k,1]
    else
        throw(ArgumentError("[k] only defined for 1 x ∞ and ∞ x 1 operators"))
    end
end


function *(A::KroneckerOperator, B::KroneckerOperator)
    dspB = domainspace(B)
    rspA = rangespace(A)
    AB = Tuple([a*b for (a,b) in zip(A.ops, B.ops)])
    KroneckerOperator(AB, dspB, rspA)
end



## Shorthand


⊗(A,B) = kron(A,B)

Base.kron(A::Operator,B::Operator) = KroneckerOperator(A,B)
Base.kron(A::Operator,B) = KroneckerOperator(A,B)
Base.kron(A,B::Operator) = KroneckerOperator(A,B)
Base.kron(A::AbstractVector{T},B::Operator) where {T<:Operator} =
    Operator{promote_type(eltype(T),eltype(B))}[kron(a,B) for a in A]
Base.kron(A::Operator,B::AbstractVector{T}) where {T<:Operator} =
    Operator{promote_type(eltype(T),eltype(A))}[kron(A,b) for b in B]
Base.kron(A::AbstractVector{T},B::UniformScaling) where {T<:Operator} =
    Operator{promote_type(eltype(T),eltype(B))}[kron(a,1.0B) for a in A]
Base.kron(A::UniformScaling,B::AbstractVector{T}) where {T<:Operator} =
    Operator{promote_type(eltype(T),eltype(A))}[kron(1.0A,b) for b in B]






## transpose


Base.transpose(K::KroneckerOperator)=KroneckerOperator(K.ops[2],K.ops[1])

for TYP in (:ConversionWrapper,:MultiplicationWrapper,:DerivativeWrapper,:IntegralWrapper,:LaplacianWrapper),
    FUNC in (:domaintensorizer,:rangetensorizer)
    @eval $FUNC(S::$TYP) = $FUNC(S.op)
end


Base.transpose(S::SpaceOperator) =
    SpaceOperator(transpose(S.op), transpose(domainspace(S)), transpose(rangespace(S)))
Base.transpose(S::ConstantTimesOperator) = sp.c*transpose(S.op)



### Calculus

function Derivative(S::TensorSpace{<:Any,<:EuclideanDomain}, order)
    @assert length(order)==length(S.spaces)
    @inline Derivative_or_I(i) = order[i]>0 ? Derivative(S.spaces[i], order[i]) : Operator(I,S.spaces[i])
    DerivativeWrapper(mapreduce(i->Derivative_or_I(i),⊗,1:length(order)), order, S)
end

function Integral(S::TensorSpace{<:Any,<:EuclideanDomain}, order)
    @assert length(order)==length(S.spaces)
    @assert max(order...)<=1
    @inline Integral_or_I(i) = order[i]>0 ? Integral(S.spaces[i]) : Operator(I,S.spaces[i])
    IntegralWrapper(mapreduce(i->Integral_or_I(i),⊗,1:length(order)))
end

DefiniteIntegral(S::TensorSpace) = DefiniteIntegralWrapper(mapreduce(DefiniteIntegral,⊗,S.spaces))
function DefiniteIntegral(S::TensorSpace, dim::Vector{Int})
    @assert length(dim)==length(S.spaces)
    @inline Int_or_I(i) = 1==dim[i] ? DefiniteIntegral(S.spaces[i]) : Operator(I,S.spaces[i])
    DefiniteIntegralWrapper(mapreduce(i->Int_or_I(i),⊗,1:length(dim)))
end



### Copy

# finds block lengths for a subrange
blocklengthrange(rt, B::Block) = [blocklength(rt,B)]
blocklengthrange(rt, B::BlockRange) = blocklength(rt,B)
function blocklengthrange(rt, kr)
    KR=block(rt,first(kr)):block(rt,last(kr))
    Klengths=Vector{Int}(length(KR))
    for ν in eachindex(KR)
        Klengths[ν]=blocklength(rt,KR[ν])
    end
    Klengths[1]+=blockstart(rt,KR[1])-kr[1]
    Klengths[end]+=kr[end]-blockstop(rt,KR[end])
    Klengths
end

function bandedblockbanded_convert!(ret, S::SubOperator, KO, rt, dt)
    pinds = parentindices(S)
    kr,jr = pinds

    kr1,jr1 = reindex(S,pinds,(1,1))

    Kshft = block(rt,kr1)-1
    Jshft = block(dt,jr1)-1



    for J=blockaxes(ret,2)
        jshft = (J==Block(1) ? jr1 : blockstart(dt,J+Jshft)) - 1
        for K=blockcolrange(ret,J)
            Bs = view(ret,K,J)
            Bspinds = parentindices(Bs)
            kshft = (K==Block(1) ? kr1 : blockstart(rt,K+Kshft)) - 1
            for ξ=1:size(Bs,2),κ=colrange(Bs,ξ)
                Bs[κ,ξ] = S[reindex(Bs,Bspinds,(κ,ξ))...]
            end
        end
    end

    ret
end



function default_BandedBlockBandedMatrix(S)
    KO = parent(S)
    rt=rangespace(KO)
    dt=domainspace(KO)
    ret = BandedBlockBandedMatrix(Zeros, S)
    bandedblockbanded_convert!(ret, S, parent(S), rt, dt)
end

BandedBlockBandedMatrix(S::SubOperator) = default_BandedBlockBandedMatrix(S)


const Trivial2DTensorizer = CachedIterator{Tuple{Int,Int},
                                             TrivialTensorizer{2}}

# This routine is an efficient version of KroneckerOperator for the case of
# tensor product of trivial blocks

function BandedBlockBandedMatrix(S::SubOperator{T,KroneckerOperator{TT,DS,RS,
                                     Trivial2DTensorizer,Trivial2DTensorizer,T},
                                     Tuple{BlockRange1,BlockRange1}}) where {TT,DS,RS,T}
    KR,JR = parentindices(S)
    KR_i, JR_i = Int.(KR), Int.(JR)

    KO=parent(S)

    ret = BandedBlockBandedMatrix(Zeros, S)

    A,B = KO.ops


    AA = strictconvert(BandedMatrix, view(A, Block(1):last(KR),Block(1):last(JR)))
    Al,Au = bandwidths(AA)
    BB = strictconvert(BandedMatrix, view(B, Block(1):last(KR),Block(1):last(JR)))
    Bl,Bu = bandwidths(BB)
    λ,μ = subblockbandwidths(ret)

    for J in blockaxes(ret,2), K in blockcolrange(ret,J)
        n,m = KR_i[Int(K)],JR_i[Int(J)]
        Bs = view(ret, K, J)
        l = min(Al,Bu+n-m,λ)
        u = min(Au,Bl+m-n,μ)
        for j=1:m, k=max(1,j-u):min(n,j+l)
            a = AA[k,j]
            b = BB[n-k+1,m-j+1]
            c = a*b
            inbands_setindex!(Bs,c,k,j)
        end
    end
    ret
end

convert(::Type{BandedBlockBandedMatrix}, S::SubOperator{T,KroneckerOperator{TT,DS,RS,
                                     Trivial2DTensorizer,Trivial2DTensorizer,T},
                                     Tuple{BlockRange1,BlockRange1}}) where {TT,DS,RS,T} =
    BandedBlockBandedMatrix(S)

## TensorSpace operators


## Conversion
# TODO: we explicetly state type to avoid type inference bug in 0.4

ConcreteConversion(a::BivariateSpace,b::BivariateSpace) =
    ConcreteConversion(promote_type(prectype(a),prectype(b)), a,b)

function Conversion(a::TensorSpace2D,b::TensorSpace2D)
    C1 = Conversion(a.spaces[1],b.spaces[1])
    C2 = Conversion(a.spaces[2],b.spaces[2])
    K = KroneckerOperator(C1, C2, a, b)
    T = promote_type(prectype(a),prectype(b))
    ConversionWrapper(strictconvert(Operator{T}, K))
end

function Conversion(a::TensorSpace,b::TensorSpace)
    C = map(Conversion,a.spaces,b.spaces)
    K = KroneckerOperator(C, a, b)
    T = promote_type(prectype(a),prectype(b))
    ConversionWrapper(strictconvert(Operator{T}, K))
end

function Multiplication(f::Fun{<:TensorSpace}, S::TensorSpace)
    lr = LowRankFun(f)
    MAs = map(a->Multiplication(a,S.spaces[1]), lr.A)
    MBs = map(a->Multiplication(a,S.spaces[2]), lr.B)
    ops = map(kron, MAs, MBs)
    MultiplicationWrapper(f, PlusOperator(ops))
end

## Functionals
Evaluation(sp::TensorSpace,x::SVector) = EvaluationWrapper(sp,x,zeros(Int,length(x)),⊗(map(Evaluation,sp.spaces,x)...))
Evaluation(sp::TensorSpace,x::Tuple) = Evaluation(sp,SVector(x...))



# it's faster to build the operators to the last b
function mul_coefficients(A::SubOperator{T,KKO,NTuple{2,UnitRange{Int}}}, b) where {T,KKO<:KroneckerOperator}
    P = parent(A)
    kr,jr = parentindices(A)
    dt,rt = domaintensorizer(P),rangetensorizer(P)
    KR,JR = Block(1):block(rt,last(kr)),Block(1):block(dt,last(jr))
    M = P[KR,JR]
    M*pad(b, size(M,2))
end

Base.getindex(A::KroneckerOperator, B::MultivariateFun) = A[Fun(B)]
function Base.getindex(K::KroneckerOperator, f::LowRankFun)
    op1, op2 = K.ops
    sum(zip(f.A, f.B)) do (A,B)
        op1[A] ⊗ op2[B]
    end
end
Base.getindex(K::KroneckerOperator, B::ProductFun) = K[LowRankFun(B)]

for F in [:MultivariateFun, :ProductFun, :LowRankFun]
    for O in WrapperList
        @eval Base.getindex(K::$O{<:KroneckerOperator}, f::$F) = K.op[f]
        @eval (*)(A::$O{<:KroneckerOperator}, B::$F) = A.op * B
        @eval (*)(A::$F, B::$O{<:KroneckerOperator}) = A * B.op
    end
end
(*)(A::KroneckerOperator, B::MultivariateFun) = A * Fun(B)
(*)(A::MultivariateFun, B::KroneckerOperator) = Fun(A) * B

(*)(ko::KroneckerOperator, pf::ProductFun) = ko * LowRankFun(pf)
(*)(pf::ProductFun, ko::KroneckerOperator) = LowRankFun(pf) * ko

# if the second operator is a constant, we may scale the first operator,
# and apply it on the coefficients
function (*)(ko::KroneckerOperator{<:Tuple{<:Operator, <:ConstantOperator}}, pf::ProductFun)
    O1, O2 = ko.ops
    O12 = O2.λ * O1
    ProductFun(map(x -> O12*x, pf.coefficients), factor(pf.space, 2))
end

function (*)(ko::KroneckerOperator, lrf::LowRankFun)
    O1, O2 = ko.ops
    sum(zip(lrf.A, lrf.B)) do (f1, f2)
        (O1*f1) ⊗ (O2*f2)
    end
end
function (*)(lrf::LowRankFun, ko::KroneckerOperator)
    O1, O2 = ko.ops
    sum(zip(lrf.A, lrf.B)) do (f1, f2)
        (f1*O1) ⊗ (f2*O2)
    end
end

_mulop(pf::ProductFun, ::BivariateSpace, O::Operator) = LowRankFun(pf) * O
_mulop(O::Operator, ::BivariateSpace, pf::ProductFun) = O * LowRankFun(pf)
(*)(P::PlusOperator, lrf::LowRankFun) = sum(op -> op*lrf, P.ops)
(*)(lrf::LowRankFun, P::PlusOperator) = sum(op -> lrf*op, P.ops)
(*)(lrf::LowRankFun, T::TimesOperator) = (lrf * T.ops[1]) * foldr(*, T.ops[2:end])
(*)(T::TimesOperator, lrf::LowRankFun) = foldr(*, T.ops[1:end-1]) * (T.ops[end] * lrf)
