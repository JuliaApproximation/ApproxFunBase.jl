

export HermitianOperator, SymmetricOperator, AdjointOperator, TransposeOperator


struct HermitianOperator{T<:Number,B<:Operator} <: Operator{T}
    op::B
    uplo::Char
end

HermitianOperator(B::Operator{T}, uplo::Symbol=:U) where {T<:Number} = HermitianOperator{T,typeof(B)}(B, char_uplo(uplo))
Operator{T}(A::HermitianOperator) where {T}=HermitianOperator(strictconvert(Operator{T},A.op), A.uplo)
function HermitianOperator{T,B}(H::HermitianOperator) where {T<:Number,B<:Operator}
    HermitianOperator{T,B}(
        strictconvert(B, H.op),
        H.uplo
        )
end

domainspace(P::HermitianOperator)=domainspace(P.op)
rangespace(P::HermitianOperator)=rangespace(P.op)
domain(P::HermitianOperator)=domain(P.op)
bandwidths(P::HermitianOperator) = (b = bandwidth(P.op, P.uplo == 'L' ? 1 : P.uplo == 'U' ? 2 : 0); (b, b))

function getindex(P::HermitianOperator,k::Integer,j::Integer)
    if P.uplo == 'L'
        if j > k
            conj(P.op[j,k])
        else
            P.op[k,j]
        end
    elseif P.uplo == 'U'
        if j < k
            conj(P.op[j,k])
        else
            P.op[k,j]
        end
    end
end

Hermitian(A::Operator, uplo::Symbol)=HermitianOperator(A, uplo)
Hermitian(A::Operator)=HermitianOperator(A)


struct SymmetricOperator{T<:Number,B<:Operator} <: Operator{T}
    op::B
    uplo::Char
end

SymmetricOperator(B::Operator{T}, uplo::Symbol=:U) where {T<:Number} = SymmetricOperator{T,typeof(B)}(B, char_uplo(uplo))
Operator{T}(A::SymmetricOperator) where {T}=SymmetricOperator(strictconvert(Operator{T},A.op), A.uplo)
function SymmetricOperator{T,B}(S::SymmetricOperator) where {T<:Number,B<:Operator}
    SymmetricOperator{T,B}(
        strictconvert(B, S.op),
        S.uplo,
        )
end

domainspace(P::SymmetricOperator)=domainspace(P.op)
rangespace(P::SymmetricOperator)=rangespace(P.op)
domain(P::SymmetricOperator)=domain(P.op)
bandwidths(P::SymmetricOperator) = (b = bandwidth(P.op, P.uplo == 'L' ? 1 : P.uplo == 'U' ? 2 : 0); (b, b))

function getindex(P::SymmetricOperator,k::Integer,j::Integer)
    if P.uplo == 'L'
        if j > k
            P.op[j,k]
        else
            P.op[k,j]
        end
    elseif P.uplo == 'U'
        if j < k
            P.op[j,k]
        else
            P.op[k,j]
        end
    end
end

Symmetric(A::Operator, uplo::Symbol)=SymmetricOperator(A, uplo)
Symmetric(A::Operator)=SymmetricOperator(A)


struct AdjointOperator{T<:Number,B<:Operator} <: Operator{T}
    op::B
end

AdjointOperator(B::Operator{T}) where {T<:Number}=AdjointOperator{T,typeof(B)}(B)
Operator{T}(A::AdjointOperator) where {T}=AdjointOperator(strictconvert(Operator{T},A.op))
function AdjointOperator{T,B}(A::AdjointOperator) where {T<:Number,B<:Operator}
    AdjointOperator{T,B}(strictconvert(B, A.op))
end

domainspace(P::AdjointOperator)=rangespace(P.op)
rangespace(P::AdjointOperator)=domainspace(P.op)
domain(P::AdjointOperator)=domain(P.op)
bandwidths(P::AdjointOperator) = reverse(bandwidths(P.op))

getindex(P::AdjointOperator,k::Integer,j::Integer) = conj(P.op[j,k])
getindex(P::AdjointOperator,inds...) = adjoint(P.op[reverse(inds)...])

function BandedMatrix(S::SubOperator{T,TO}) where {T,TO<:AdjointOperator}
    kr,jr=parentindices(S)
    adjoint(BandedMatrix(view(parent(S).op,jr,kr)))
end

adjoint(A::Operator)=AdjointOperator(A)


struct TransposeOperator{T<:Number,B<:Operator} <: Operator{T}
    op::B
end

TransposeOperator(B::Operator{T}) where {T<:Number}=TransposeOperator{T,typeof(B)}(B)
Operator{T}(A::TransposeOperator) where {T}=TransposeOperator(strictconvert(Operator{T},A.op))
function TransposeOperator{T,B}(A::TransposeOperator) where {T<:Number,B<:Operator}
    TransposeOperator{T,B}(strictconvert(B, A.op))
end

domainspace(P::TransposeOperator)=rangespace(P.op)
rangespace(P::TransposeOperator)=domainspace(P.op)
domain(P::TransposeOperator)=domain(P.op)
bandwidths(P::TransposeOperator) = reverse(bandwidths(P.op))

getindex(P::TransposeOperator,k::Integer,j::Integer) = P.op[j,k]
getindex(P::TransposeOperator,inds...) = transpose(P.op[reverse(inds)...])

function BandedMatrix(S::SubOperator{T,TO}) where {T,TO<:TransposeOperator}
    kr,jr=parentindices(S)
    transpose(BandedMatrix(view(parent(S).op,jr,kr)))
end

transpose(A::Operator)=TransposeOperator(A)
