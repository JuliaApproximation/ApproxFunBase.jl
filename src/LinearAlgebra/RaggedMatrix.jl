# FiniteRange gives the nonzero entries in a row/column
struct FiniteRange end

getindex(A::AbstractMatrix,::Type{FiniteRange},j::Integer) = A[colrange(A,j),j]
getindex(A::AbstractMatrix,k::Integer,::Type{FiniteRange}) = A[k,1:rowstop(A,k)]

const ⤓ = FiniteRange

mutable struct RaggedMatrix{T} <: AbstractMatrix{T}
    data::Vector{T} # a Vector of non-zero entries
    cols::Vector{Int} # a Vector specifying the first index of each column
    m::Int #Number of rows
    function RaggedMatrix{T}(data::Vector{T}, cols::Vector{Int}, m::Int) where T
        # make sure the cols are monitonically increasing
        @assert 1==cols[1]
        for j=1:length(cols)-1
            @assert cols[j] ≤ cols[j+1]
        end
        @assert cols[end] == length(data)+1

        # make sure we have less entries than the size of the matrix
        @assert length(data) ≤ m*(length(cols)-1)

        new{T}(data,cols,m)
    end
end

RaggedMatrix(dat::Vector,cols::Vector{Int},m::Int) =
    RaggedMatrix{eltype(dat)}(dat,cols,m)

RaggedMatrix{T}(::UndefInitializer, m::Int, colns::AbstractVector{Int}) where {T} =
    RaggedMatrix(Vector{T}(undef, sum(colns)),Int[1;1 .+ cumsum(colns)],m)


size(A::RaggedMatrix) = (A.m,length(A.cols)-1)

colstart(A::RaggedMatrix,j::Integer) = 1
colstop(A::RaggedMatrix,j::Integer) = min(A.cols[j+1]-A.cols[j],size(A,1))

Base.@propagate_inbounds function incol(A, k, j, ind = A.cols[j]+k-1)
    ind < A.cols[j+1]
end

Base.@propagate_inbounds function incols_getindex(A::RaggedMatrix, k::Int, j::Int, ind = A.cols[j]+k-1)
    A.data[ind]
end
Base.@propagate_inbounds function incols_setindex!(A::RaggedMatrix, v, k::Int, j::Int, ind = A.cols[j]+k-1)
    A.data[ind] = v
    A
end

Base.@propagate_inbounds function getindex(A::RaggedMatrix,k::Int,j::Int)
    @boundscheck if k>size(A,1) || k < 1 || j>size(A,2) || j < 1
        throw(BoundsError(A,(k,j)))
    end

    ind = A.cols[j]+k-1
    if incol(A, k, j, ind)
        incols_getindex(A, k, j, ind)
    else
        zero(eltype(A))
    end
end

Base.@propagate_inbounds function setindex!(A::RaggedMatrix,v,k::Int,j::Int)
    @boundscheck if k>size(A,1) || k < 1 || j>size(A,2) || j < 1
        throw(BoundsError(A,(k,j)))
    end

    ind = A.cols[j]+k-1
    if incol(A, k, j, ind)
        incols_setindex!(A, v, k, j, ind)
    elseif v ≠ 0
        throw(ArgumentError("Can't set index $((k,j)) of a RaggedMatrix to a non-zero value"))
    end
    A
end

convert(::Type{RaggedMatrix{T}}, R::RaggedMatrix{T}) where T = R

convert(::Type{RaggedMatrix{T}}, R::RaggedMatrix) where T =
    RaggedMatrix{T}(Vector{T}(R.data), R.cols, R.m)


function convert(::Type{Matrix{T}}, A::RaggedMatrix) where T
    ret = zeros(T, size(A))
    @inbounds for j in axes(A,2), k in colrange(A,j)
        v = incols_getindex(A, k, j)
        ret[k,j] = v
    end
    ret
end

convert(::Type{Matrix}, A::RaggedMatrix) = Matrix{eltype(A)}(A)

convert(::Type{RaggedMatrix}, B::BandedMatrix) = RaggedMatrix{eltype(B)}(B)

function convert(::Type{RaggedMatrix{T}}, B::AbstractMatrix) where T
    ret = RaggedMatrix(Zeros{T}(size(B)), Int[colstop(B,j) for j=axes(B,2)])
    @inbounds for j in axes(B,2), k in colrange(B,j)
        incols_setindex!(ret, B[k,j], k, j)
    end
    ret
end

convert(::Type{RaggedMatrix}, B::AbstractMatrix) = strictconvert(RaggedMatrix{eltype(B)}, B)

RaggedMatrix(B::AbstractMatrix) = strictconvert(RaggedMatrix, B)
RaggedMatrix{T}(B::AbstractMatrix) where T = strictconvert(RaggedMatrix{T}, B)

similar(B::RaggedMatrix,::Type{T}) where {T} = RaggedMatrix(similar(B.data, T),copy(B.cols),B.m)

for (op,bop) in ((:(Base.rand), :rrand),)
    @eval begin
        $bop(::Type{T}, m::Int, colns::AbstractVector{Int}) where {T} =
            RaggedMatrix($op(T,sum(colns)),[1; (1 .+ cumsum(colns))],m)
        $bop(m::Int, colns::AbstractVector{Int}) = $bop(Float64, m, colns)
    end
end

function RaggedMatrix{T}(Z::Zeros, colns::AbstractVector{Int}) where {T}
    if size(Z,2) ≠ length(colns)
        throw(DimensionMismatch())
    end
    RaggedMatrix(zeros(T,sum(colns)), [1; (1 .+cumsum(colns))], size(Z,1))
end

function RaggedMatrix{T}(A::AbstractMatrix, colns::AbstractVector{Int}) where T
    Base.require_one_based_indexing(A)
    Base.require_one_based_indexing(colns)
    ret = RaggedMatrix{T}(undef, size(A,1), colns)
    (length(colns) == size(A,2) && all(<=(size(A,1)), colns)) ||
        throw(ArgumentError("column stops $colns incompatible with input matrix of size $(size(A))"))
    for j in axes(A,2), k = 1:colns[j]
        @inbounds incols_setindex!(ret, A[k,j], k, j)
    end
    ret
end

RaggedMatrix(A::AbstractMatrix, colns::AbstractVector{Int}) = RaggedMatrix{eltype(A)}(A, colns)

function Base.replace_in_print_matrix(A::RaggedMatrix,i::Integer,j::Integer,s::AbstractString)
    incol(A, i, j) ? s : Base.replace_with_centered_mark(s)
end

## BLAS

function mul!(y::Vector, A::RaggedMatrix, b::Vector)
    m=size(A,2)

    if m ≠ length(b) || size(A,1) ≠ length(y)
        throw(BoundsError())
    end
    T=eltype(y)
    fill!(y,zero(T))
    for j in axes(A,2)
        kr=A.cols[j]:A.cols[j+1]-1
        axpy!(b[j],view(A.data,kr),view(y,1:length(kr)))
    end
    y
end


function axpy!(a, X::RaggedMatrix, Y::RaggedMatrix)
    if size(X) ≠ size(Y)
        throw(BoundsError())
    end

    if X.cols == Y.cols
        axpy!(a,X.data,Y.data)
    else
        for j = axes(X,2)
            Xn = colstop(X,j)
            Yn = colstop(Y,j)
            if Xn > Yn  # check zeros otherwise
                for k = Yn+1:Xn
                    @assert iszero(X[k,j])
                end
            end
            cs = min(Xn,Yn)
            axpy!(a, view(X.data, range(X.cols[j], length=cs)),
                     view(Y.data, range(Y.cols[j], length=cs)))
        end
    end
    Y
end

colstop(X::SubArray{T,2,RaggedMatrix{T},NTuple{2,UnitRange{Int}}},
     j::Integer) where {T} = min(colstop(parent(X),j + first(parentindices(X)[2])-1) -
                                            first(parentindices(X)[1]) + 1,
                            size(X,1))

function axpy!(a,X::RaggedMatrix,
                    Y::SubArray{T,2,RaggedMatrix{T},NTuple{2,UnitRange{Int}}}) where T
    if size(X) ≠ size(Y)
        throw(BoundsError())
    end

    P = parent(Y)
    ksh = first(parentindices(Y)[1]) - 1  # how much to shift
    jsh = first(parentindices(Y)[2]) - 1  # how much to shift

    for j=axes(X,2)
        cx=colstop(X,j)
        cy=colstop(Y,j)
        if cx > cy
            for k=cy+1:cx
                if X[k,j] ≠ 0
                    throw(BoundsError("Trying to add a non-zero to a zero."))
                end
            end
            kr = range(X.cols[j], length=cy)
        else
            kr = X.cols[j]:X.cols[j+1]-1
        end


        axpy!(a,view(X.data,kr),
                    view(P.data,(P.cols[j + jsh] + ksh-1) .+ (1:length(kr))))
    end

    Y
end


function *(A::RaggedMatrix,B::RaggedMatrix)
    cols = zeros(Int,size(B,2))
    T = promote_type(eltype(A),eltype(B))
    for j=axes(B,2),k=colrange(B,j)
        cols[j] = max(cols[j],colstop(A,k))
    end

    unsafe_mul!(RaggedMatrix{T}(undef, size(A,1), cols), A, B)
end

function unsafe_mul!(Y::RaggedMatrix,A::RaggedMatrix,B::RaggedMatrix)
    fill!(Y.data,0)

    for j=axes(B,2),k=colrange(B,j)
        axpy!(B[k,j], view(A,colrange(A,k),k),
            view(Y.data,Y.cols[j] .- 1 .+ (colrange(A,k))))
    end

    Y
end

function mul!(Y::RaggedMatrix,A::RaggedMatrix,B::RaggedMatrix)
    for j=axes(B,2)
        col = 0
        for k=colrange(B,j)
            col = max(col,colstop(A,k))
        end

        if col > colstop(Y,j)
            throw(BoundsError(Y, (col,j)))
        end
    end

    unsafe_mul!(Y,A,B)
end
