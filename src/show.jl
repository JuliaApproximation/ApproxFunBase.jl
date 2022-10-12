## Fun

show(io::IO, ::MIME"text/plain", f::Fun) = show(io, f)

function show(io::IO, f::Fun)
    print(io,"Fun(")
    show(io,space(f))
    print(io,", ")
    show(io,coefficients(f))
    print(io,")")
end

show(io::IO,f::Fun{<:ConstantSpace{AnyDomain}}) =
    print(io, convert(Number,f), " anywhere")

show(io::IO,f::Fun{<:ConstantSpace}) =
    print(io, convert(Number,f), " on ", domain(f))

## MultivariateFun

show(io::IO, ::MIME"text/plain", f::MultivariateFun) = show(io, f)

function show(io::IO,L::LowRankFun)
    print(io,"LowRankFun on ",space(L)," of rank ",rank(L),".")
end

function show(io::IO,P::ProductFun)
    print(io,"ProductFun on ",space(P))
end

## Operator

summarystr(B::Operator) = string(typeof(B).name.name, " : ", domainspace(B), " → ", rangespace(B))
summary(io::IO, B::Operator) = print(io, summarystr(B))

struct PrintShow
    c::Char
end
Base.show(io::IO, N::PrintShow) = print(io, N.c)

show(io::IO, B::Operator; kw...) = summary(io, B)

function show(io::IO, mimetype::MIME"text/plain", @nospecialize(B::Operator); header::Bool=true)
    header && summary(io, B)
    dsp = domainspace(B)

    sz1_B, sz2_B = size(B)

    iocompact = haskey(io, :compact) ? io : IOContext(io, :compact => true)

    if !isambiguous(domainspace(B)) && (eltype(B) <: Number)
        println(io)
        if isbanded(B) && isinf(sz1_B) && isinf(sz2_B)
            BM=B[1:10,1:10]

            M=Matrix{Union{eltype(B), PrintShow}}(undef,11,11)
            fill!(M,PrintShow('⋅'))
            for j = 1:size(BM,2),k = colrange(BM,j)
                M[k,j]=BM[k,j]
            end

            for k=max(1,11-bandwidth(B,2)):11
                M[k,end]=PrintShow('⋱')
            end
            for j=max(1,11-bandwidth(B,1)):10
                M[end,j]=PrintShow('⋱')
            end

            print_array(iocompact, M)
        elseif isinf(sz1_B) && isinf(sz2_B)
            BM=B[1:10,1:10]

            M=Matrix{Union{eltype(B), PrintShow}}(undef,11,11)
            for I in CartesianIndices(axes(BM))
                M[I]=BM[Tuple(I)...] # not certain if indexing with CartesianIndex is implemented
            end

            M[1,end]=PrintShow('⋯')
            M[end,1]=PrintShow('⋮')

            for k=2:11
                M[k,end]=PrintShow('⋱')
            end
            for k=2:11
                M[end,k]=PrintShow('⋱')
            end

            print_array(iocompact, M)
        elseif isinf(sz1_B)
            sz2int = Int(sz2_B)::Int
            BM=B[1:10,1:sz2int]

            M=Matrix{Union{eltype(B), PrintShow}}(undef,11,sz2int)
            for I in CartesianIndices(axes(BM))
                M[I]=BM[Tuple(I)...]
            end
            for k=1:sz2int
                M[end,k]=PrintShow('⋮')
            end

            print_array(iocompact, M)
        elseif isinf(sz2_B)
            sz1int = Int(sz1_B)::Int
            BM=B[1:sz1int,1:10]

            M=Matrix{Union{eltype(B), PrintShow}}(undef,sz1int,11)
            for I in CartesianIndices(axes(BM))
                M[I]=BM[Tuple(I)...]
            end
            for k=1:sz1int
                M[k,end]=PrintShow('⋯')
            end

            print_array(iocompact, M)
        else
            sz1int = Int(sz1_B)::Int
            sz2int = Int(sz2_B)::Int
            print_array(iocompact, AbstractMatrix(B)[1:sz1int,1:sz2int])
        end
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
        print(io,"⊕")
        show(io,sp)
    end
end


function show(io::IO,ss::PiecewiseSpace)
    s = components(ss)
    show(io,s[1])
    for sp in s[2:end]
        print(io,"⨄")
        show(io,sp)
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
