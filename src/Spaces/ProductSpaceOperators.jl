export continuity

## Space promotion for InterlaceOperator
# It's here because we need DirectSumSpace

for TYP in (:PiecewiseSpace,:ArraySpace)
    @eval begin
        function promotedomainspace(A::InterlaceOperator{T,2},sp::$TYP) where T
            if domainspace(A) == sp
                return A
            end
            @assert size(A.ops,2) == length(sp)
            InterlaceOperator([promotedomainspace(A.ops[k,j],sp[j]) for k=1:size(A.ops,1),j=1:size(A.ops,2)],$TYP)
        end
        function interlace_choosedomainspace(ops,rs::$TYP)
            @assert length(ops) == length(rs)
            # this ensures correct dispatch for union
            sps = Array{Space}(
                filter(x->!isambiguous(x),map((op,s)->choosedomainspace(op,s),ops,rs)))
            if isempty(sps)
                UnsetSpace()
            else
                union(sps...)
            end
        end
    end
end




function continuity(sp::PiecewiseSpace,order::Integer)
    m=ncomponents(sp)
    B=zeros(Operator{prectype(sp)},m-1,m)

    for k=1:m-1
        B[k,k] = Evaluation(component(sp,k),rightendpoint,order)
        B[k,k+1] = -Evaluation(component(sp,k+1),leftendpoint,order)
    end

    InterlaceOperator(B,PiecewiseSpace,ArraySpace)
end

function continuity(sp::PiecewiseSpace,kr::UnitRange)
    @assert first(kr)==0
    m=ncomponents(sp)
    B=zeros(Operator{prectype(sp)},length(kr)*(m-1),m)
    for r in kr
        B[(m-1)*r+1:(m-1)*(r+1),:] = continuity(sp,r).ops
    end
    InterlaceOperator(B,PiecewiseSpace,ArraySpace)
end


continuity(d::UnionDomain,k) = continuity(Space(d),k)
continuity(d) = continuity(d,0)

# TODO: general wrappers

Evaluation(S::SumSpace,x,order) =
    EvaluationWrapper(S,x,order,
        InterlaceOperator(RowVector(vnocat(map(s->Evaluation(s,x,order),components(S))...)),SumSpace))


ToeplitzOperator(G::Fun{<:MatrixSpace}) = interlace(map(ToeplitzOperator,Array(G)))

## Sum Space




## Conversion

function coefficients(v::AbstractVector,a::ArraySpace,b::ArraySpace)
    if a==b
        v
    else
        interlace(map((f,s)->Fun(f,s),Fun(a,v),b),b)
    end
end


# ArraySpace is straight forward

function Conversion(a::ArraySpace, b::ArraySpace)
    @assert size(a) == size(b)
    ConversionWrapper(InterlaceOperator(Diagonal(Conversion.(vec(a.spaces), vec(b.spaces))), a, b))
end


# Sum Space and PiecewiseSpace need to allow permutation of space orders
for TYP in (:SumSpace,:PiecewiseSpace)
    @eval function Conversion(S1::$TYP,S2::$TYP)
        v1 = collect(S1.spaces)
        v2 = collect(S2.spaces)

        sort1 = sort(collect(v1))
        sort2 = sort(collect(v2))

        T = promote_type(eltype(domain(S1)),eltype(domain(S2)))

        if any(s->!isinf(dimension(s)),v1) || any(s->!isinf(dimension(s)),v2)
            @assert length(S1.spaces) == length(S2.spaces) == 2
            if hasconversion(S1, S2.spaces[1])
                ops = Operator{T}[Conversion(S1.spaces[1], S2.spaces[1]) Conversion(S1.spaces[2], S2.spaces[1]);
                                ZeroOperator(T, S1.spaces[1], S2.spaces[2]) ZeroOperator(T, S1.spaces[2], S2.spaces[2])]
            elseif hasconversion(S1, S2.spaces[2])
                ops = Operator{T}[ZeroOperator(T, S1.spaces[1], S2.spaces[1]) ZeroOperator(T, S1.spaces[2], S2.spaces[1]);
                                  Conversion(S1.spaces[1], S2.spaces[2]) Conversion(S1.spaces[2], S2.spaces[2])]
            else
                error("Not implemented")
            end
            ConversionWrapper(InterlaceOperator(ops, S1, S2))
        elseif sort1 == sort2
            # swaps sumspace order
            ConversionWrapper(PermutationOperator{T}(perm(v1,v2),S1,S2))
        elseif all(map(hasconversion,v1,v2))
            # we can block convert
            ConversionWrapper(SpaceOperator(
                InterlaceOperator(Diagonal([map(Conversion,v1,v2)...]),$TYP),
                S1,S2))
        elseif all(map(hasconversion,sort1,sort2))
            # we can block convert
            P1 = PermutationOperator{T}(perm(v1,sort1),S1,$TYP(sort1))
            P2 = PermutationOperator{T}(perm(sort2,v2),$TYP(sort2),S2)
            ConversionWrapper(TimesOperator(
                [P2,InterlaceOperator(Diagonal([map(Conversion,sort1,sort2)...]),$TYP),P1]))
        elseif map(canonicalspace,S1.spaces) == map(canonicalspace,S2.spaces)
            error("Not implemented")
        else
            # try sorting canonicalspace
            csort1 = sort(collect(map(canonicalspace,v1)))
            csort2 = sort(collect(map(canonicalspace,v2)))
            if csort1 == csort2
                # we can block convert after permuting
                prm = perm(map(canonicalspace,v1),map(canonicalspace,v2))
                ds2 = $TYP(S1.spaces[prm])
                P = PermutationOperator{T}(prm, S1, ds2)
                ConversionWrapper(TimesOperator(Conversion(ds2,S2),P))
            elseif all(map(hasconversion,csort1,csort2))
                C1 = Conversion(S1,$TYP(csort1))
                C2 = Conversion($TYP(csort1),$TYP(csort2))
                C3 = Conversion($TYP(csort2),S2)

                ConversionWrapper(TimesOperator([C3,C2,C1]))
            else
                # we don't know how to convert so go to default
                defaultConversion(S1,S2)
            end
        end
    end
end

hasconversion(a::SumSpace, b::Space) = all(s -> hasconversion(s, b), a.spaces)

function Conversion(a::SumSpace, b::Space)
    if !hasconversion(a, b)
        throw(ArgumentError("Cannot convert $a to $b"))
    end

    m=zeros(Operator{promote_type(prectype(a), prectype(b))},1,length(a.spaces))
    ops = map(x -> Conversion(x,b), a.spaces)
    copyto!(m, ops)
    bbw = bandwidthsmax(ops, blockbandwidths)
    irb = any(israggedbelow, ops)
    return ConversionWrapper(
        InterlaceOperator(m, a, b,
            cache(interlacer(a)),
            cache(BlockInterlacer((Fill(1,∞),))),
            (1-dimension(b),dimension(a)-1),
            bbw,
            irb,
        )
    )
end



for (OPrule,OP) in ((:conversion_rule,:conversion_type),(:maxspace_rule,:maxspace),
                        (:union_rule,:union))
    for TYP in (:SumSpace,:PiecewiseSpace)
        @eval function $OPrule(S1sp::$TYP,S2sp::$TYP)
            S1 = components(S1sp)
            S2 = components(S2sp)
            cs1,cs2=map(canonicalspace,S1),map(canonicalspace,S2)
            if length(S1) != length(S2)
                NoSpace()
            elseif canonicalspace(S1sp) == canonicalspace(S2sp)  # this sorts S1 and S2
                S1sp ≤ S2sp ? S1sp : S2sp  # choose smallest space by sorting
            elseif cs1 == cs2
                # we can just map down
                # $TYP(map($OP,S1.spaces,S2.spaces))
                # this is commented out due to Issue #13261
                newspaces = [$OP(S1[k],S2[k]) for k=1:length(S1)]
                if any(b->b==NoSpace(),newspaces)
                    NoSpace()
                else
                    $TYP(newspaces)
                end
            elseif sort(collect(cs1)) == sort(collect(cs2))
                # sort S1
                p=perm(cs1,cs2)
                $OP($TYP(S1[p]),S2sp)
            elseif length(S1) == length(S2) == 2  &&
                    $OP(S1[1],S2[1]) != NoSpace() &&
                    $OP(S1[2],S2[2]) != NoSpace()
                #TODO: general length
                $TYP($OP(S1[1],S2[1]),
                     $OP(S1[2],S2[2]))
            elseif length(S1) == length(S2) == 2  &&
                    $OP(S1[1],S2[2])!=NoSpace() &&
                    $OP(S1[2],S2[1])!=NoSpace()
                #TODO: general length
                $TYP($OP(S1[1],S2[2]),
                     $OP(S1[2],S2[1]))
            else
                NoSpace()
            end
        end
    end
end




## Derivative

#TODO: do in @calculus_operator?

_spacename(::SumSpace) = SumSpace
_spacename(::PiecewiseSpace) = PiecewiseSpace

@inline function InterlaceOperator_Diagonal(t, ds, rs = _spacename(ds)(map(rangespace, t)))
    allbanded = all(isbanded, t)
    opbw = map(bandwidths, t)
    D = Diagonal(convert_vector_or_svector(t))
    iopbw = interlace_bandwidths(D, ds, rs, allbanded, opbw)
    iopbbw = bandwidthsmax(t, blockbandwidths)
    irb = all(israggedbelow, t)
    InterlaceOperator(D, ds, rs,
        bandwidths = iopbw, blockbandwidths = iopbbw, israggedbelow = irb)
end

for (Op,OpWrap) in ((:Derivative,:DerivativeWrapper),(:Integral,:IntegralWrapper))
    @eval begin
        Base.@constprop :aggressive function $Op(S::PiecewiseSpace, k::Number)
            assert_integer(k)
            t = map(s->$Op(s,k),components(S))
            O = InterlaceOperator_Diagonal(t, S)
            $OpWrap(O,k)
        end
        Base.@constprop :aggressive function $Op(S::ArraySpace, k::Number)
            assert_integer(k)
            ops = map(s->$Op(s,k),S)
            RS = ArraySpace(reshape(map(rangespace, ops), size(S)))
            O = InterlaceOperator(Diagonal(ops), S, RS)
            $OpWrap(O,k)
        end
    end
end

Base.@constprop :aggressive function Derivative(S::SumSpace, k::Number)
    assert_integer(k)
    # we want to map before we decompose, as the map may introduce
    # mixed bases.
    if typeof(canonicaldomain(S))==typeof(domain(S))
        t = map(s->Derivative(s,k),components(S))
        O = InterlaceOperator_Diagonal(t, S)
        DerivativeWrapper(O,k)
    else
        DefaultDerivative(S,k)
    end
end

choosedomainspace(M::CalculusOperator{UnsetSpace}, sp::SumSpace) =
    mapreduce(s->choosedomainspace(M,s),union,sp.spaces)

## Multiplcation for Array*Vector

function Multiplication(f::Fun{<:MatrixSpace}, sp::VectorSpace)
    @assert size(space(f),2)==length(sp)
    m=Array(f)
    MultiplicationWrapper(f, interlace(
            Operator{promote_type(cfstype(f),prectype(sp))}[
                Multiplication(m[k,j],sp[j]) for k=1:size(m,1),j=1:size(m,2)]
        )
    )
end




## Multiply components

function Multiplication(f::Fun{<:PiecewiseSpace}, sp::PiecewiseSpace)
    p=perm(domain(f).domains,domain(sp).domains)  # sort f
    vf=components(f)[p]
    t = map(Multiplication,vf,sp.spaces)
    D = Diagonal(convert_vector_or_svector(t))
    O = InterlaceOperator(D, PiecewiseSpace)
    MultiplicationWrapper(f, O)
end

Multiplication(f::Fun{SumSpace{SV1,D,R1}},sp::SumSpace{SV2,D,R2}) where {SV1,SV2,D,R1,R2} =
    MultiplicationWrapper(f,mapreduce(g->Multiplication(g,sp),+,components(f)))

function Multiplication(f::Fun, sp::SumSpace)
    t = map(s->Multiplication(f,s),components(sp))
    O = InterlaceOperator_Diagonal(t, sp)
    MultiplicationWrapper(f, O)
end

Multiplication(f::Fun, sp::PiecewiseSpace) = MultiplicationWrapper(f, Multiplication(Fun(f,sp),sp))


# we override coefficienttimes to split the multiplication down to components as union may combine spaes

coefficienttimes(f::Fun{S1},g::Fun{S2}) where {S1<:SumSpace,S2<:SumSpace} = mapreduce(ff->ff*g,+,components(f))
coefficienttimes(f::Fun{S1},g::Fun) where {S1<:SumSpace} = mapreduce(ff->ff*g,+,components(f))
coefficienttimes(f::Fun,g::Fun{S2}) where {S2<:SumSpace} = mapreduce(gg->f*gg,+,components(g))


coefficienttimes(f::Fun{S1},g::Fun{S2}) where {S1<:PiecewiseSpace,S2<:PiecewiseSpace} =
    Fun(map(coefficienttimes,components(f),components(g)),PiecewiseSpace)


## Definite Integral

# This makes sure that the defaults from a given Domain are respected for the UnionDomain.

DefiniteIntegral(d::UnionDomain) =
    DefiniteIntegral(PiecewiseSpace(map(domainspace,map(DefiniteIntegral,d.domains))))
DefiniteLineIntegral(d::UnionDomain) =
    DefiniteLineIntegral(PiecewiseSpace(map(domainspace,map(DefiniteLineIntegral,d.domains))))


DefiniteIntegral(sp::PiecewiseSpace) =
    DefiniteIntegralWrapper(InterlaceOperator(hnocat(map(DefiniteIntegral,sp.spaces)...),sp,ConstantSpace(rangetype(sp))))
DefiniteLineIntegral(sp::PiecewiseSpace) =
    DefiniteLineIntegralWrapper(InterlaceOperator(hnocat(map(DefiniteLineIntegral,sp.spaces)...),sp,ConstantSpace(rangetype(sp))))

## TensorSpace of two PiecewiseSpaces

getindex(d::TensorSpace{Tuple{PWS1,PWS2}},i::Integer,j::Integer) where {PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace} =
    d[1][i]⊗d[2][j]
getindex(d::TensorSpace{Tuple{PWS1,PWS2}},i::AbstractRange,j::AbstractRange) where {PWS1<:PiecewiseSpace,PWS2<:PiecewiseSpace} =
    PiecewiseSpace(d[1][i])⊗PiecewiseSpace(d[2][j])

## ProductFun

##  Piecewise

function components(U::ProductFun{PS}) where PS<:PiecewiseSpace
    ps=space(U,1)
    sp2=space(U,2)
    m=length(ps)
    C=coefficients(U)
    [ProductFun(C[k:m:end,:],component(ps,k),sp2) for k=1:m]
end
