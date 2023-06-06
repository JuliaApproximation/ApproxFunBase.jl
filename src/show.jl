## Fun

show(io::IO, ::MIME"text/plain", f::Fun) = show(io, f)

function show(io::IO, f::Fun)
    print(io,"Fun(")
    show(io,space(f))
    print(io,", ")
    show(IOContext(io, :compact=>true), coefficients(f))
    print(io,")")
end

evalconst(f, ::AnyDomain) = f(0.0)
evalconst(f, d) = f(leftendpoint(d))
evalconst(f, d::Union{Point, UnionDomain{<:Any, <:Tuple{Point, Vararg{Point}}}}) = f(d)

function show(io::IO,f::Fun{<:Union{ConstantSpace, ArraySpace{<:ConstantSpace}}})
    d = domain(f)
    print(io, evalconst(f, domain(f)))
    print(io, d isa AnyDomain ? " anywhere" : " on " * string(d))
end

## MultivariateFun

show(io::IO, ::MIME"text/plain", f::MultivariateFun) = show(io, f)

function show(io::IO,L::LowRankFun)
    print(io,"LowRankFun on ",space(L)," of rank ",rank(L),".")
end

function show(io::IO,P::ProductFun)
    print(io,"ProductFun on ",space(P))
end

## Operator

summarystr(B::Operator) = string(nameof(typeof(B)), " : ", domainspace(B), " → ", rangespace(B))
summary(io::IO, B::Operator) = print(io, summarystr(B))

struct PrintShow
    c::Char
end
Base.show(io::IO, N::PrintShow) = print(io, N.c)

show(io::IO, B::Operator; kw...) = summary(io, B)

struct CharLinedMatrix{T,A<:AbstractMatrix{T}} <: AbstractMatrix{Union{T,PrintShow}}
    arr :: A
    sz :: NTuple{2,Bool}
end
Base.size(A::CharLinedMatrix) = size(A.arr) .+ A.sz

function Base.getindex(C::CharLinedMatrix, k::Int, j::Int)
    @boundscheck checkbounds(C, k, j)
    BM = C.arr
    if j in axes(BM,2) && k in axes(BM,1)
        return C.arr[k,j]
    end
    sz1, sz2 = size(C)
    if isbanded(BM) && all(C.sz)
        bw1, bw2 = bandwidths(BM)
        if k in max(1,sz1-bw2):sz1+min(0,bw1) && j == sz2
            PrintShow('⋱')
        elseif k == sz1 && j in max(1,sz2-bw1):sz2+min(0,bw2)
            PrintShow('⋱')
        else
            PrintShow('⋅')
        end
    elseif all(C.sz)
        if k == 1 && j == sz2
            PrintShow('⋯')
        elseif k in 2:sz1 && j == sz2
            PrintShow('⋱')
        elseif k == sz1 && j == 1
            PrintShow('⋮')
        elseif k == sz1 && j in 2:sz2
            PrintShow('⋱')
        end
    elseif C.sz[1]
        if k == sz1 && j in 1:sz2
            PrintShow('⋮')
        end
    elseif C.sz[2]
        if k in 1:sz1 && j == sz2
            PrintShow('⋯')
        end
    end
end

function Base.replace_in_print_matrix(C::CharLinedMatrix, k::Integer, j::Integer, s::AbstractString)
    if CartesianIndex(k,j) in CartesianIndices(C.arr)
        Base.replace_in_print_matrix(C.arr, k, j, s)
    else
        s
    end
end

function show(io::IO, mimetype::MIME"text/plain", @nospecialize(B::Operator); header::Bool=true)
    header && summary(io, B)
    dsp = domainspace(B)

    sz1_B, sz2_B = size(B)

    iocompact = haskey(io, :compact) ? io : IOContext(io, :compact => true)

    if !isambiguous(domainspace(B)) && eltype(B) <: Number
        println(io)
        sz1 = min(size(B,1),10)::Int
        sz2 = min(size(B,2),10)::Int
        C = CharLinedMatrix(B[1:sz1, 1:sz2], isinf.(size(B)))
        print_array(iocompact, C)
    end
end

## Space

function show(io::IO, S::PointSpace)
    print(io, "PointSpace(")
    show(io, points(S))
    print(io,")")
end

function show(io::IO,s::QuotientSpace)
    show(io,s.space)
    print(io," /\n")
    show(io,s.bcs)
end

function show(io::IO, m::MIME"text/plain", s::QuotientSpace)
    show(io,s.space)
    print(io," /\n")
    show(io, m, s.bcs, header = false)
end


function show(io::IO,ss::SumSpace)
    s = components(ss)
    show(io,s[1])
    for sp in s[2:end]
        print(io," ⊕ ")
        show(io,sp)
    end
end


function show(io::IO,ss::PiecewiseSpace)
    s = components(ss)
    if length(s) == 1
        print(io, "PiecewiseSpace(")
    end
    show(io,s[1])
    for sp in s[2:end]
        print(io," ⨄ ")
        show(io,sp)
    end
    if length(s) == 1
        print(io, ")")
    end
end

summarystr(ss::ArraySpace) = string(Base.dims2string(length.(axes(ss))), " ArraySpace")
summary(io::IO, ss::ArraySpace) = print(io, summarystr(ss))
function show(io::IO,ss::ArraySpace;header::Bool=true)
    header && print(io,summarystr(ss)*":\n")
    show(io, ss.spaces)
end

function show(io::IO,s::TensorSpace)
    d = length(s.spaces)
    for i=1:d-1
        show(io,s.spaces[i])
        print(io," ⊗ ")
    end
    show(io,s.spaces[d])
end

function show(io::IO,s::SubSpace)
    print(io,s.space)
    print(io,"|")
    show(io,s.indexes)
end

show(io::IO,::ConstantSpace{AnyDomain}) = print(io,"ConstantSpace")
show(io::IO,S::ConstantSpace) = print(io,"ConstantSpace($(domain(S)))")

## Segment

show(io::IO,d::Segment) = print(io,"the segment [$(leftendpoint(d)),$(rightendpoint(d))]")
