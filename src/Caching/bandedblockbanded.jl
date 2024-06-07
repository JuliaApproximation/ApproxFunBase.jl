
function CachedOperator(::Type{BandedBlockBandedMatrix}, op::Operator)
    lu = blockbandwidths(op)
    λμ = subblockbandwidths(op)
    # working on the tuples directly instead of the components helps with type-stability
    data = BandedBlockBandedMatrix{eltype(op)}(undef,
        blocklengths(rangespace(op))[1:0],blocklengths(domainspace(op))[1:0],
        lu, λμ)

    CachedOperator(op,data,size(data),domainspace(op),rangespace(op),lu .* (-1,1),false)
end

# Grow cached operator
#
function resizedata!(B::CachedOperator{T,<:BandedBlockBandedMatrix{T}}, ::Colon, col::Integer) where {T<:Number}
    if col > size(B,2)
        throw(ArgumentError("Cannot resize beyound size of operator"))
    end

    if col > B.datasize[2]
        l,u,λ,μ = B.data.l,B.data.u,B.data.λ,B.data.μ
        J = Int(block(domainspace(B),col))

        rows = blocklengths(rangespace(B.op))[1:J+l]
        cols = blocklengths(domainspace(B.op))[1:J]

        # B.data = _BandedBlockBandedMatrix(BlockedArray

        blocks = BlockedArray(pad(B.data.data.blocks,:,(l+u+1)*sum(cols)), (axes(B.data.data,1), blockedrange(cols)))
        B.data = _BandedBlockBandedMatrix(blocks, rows, cols, (l, u), (λ, μ))

        jr=B.datasize[2]+1:col
        kr=colstart(B.data,jr[1]):colstop(B.data,jr[end])

        isempty(kr) || axpy!(1.0,view(B.op,kr,jr),view(B.data,kr,jr))

        B.datasize = (last(kr),col)
    end

    B
end

function resizedata!(B::CachedOperator{T,<:BandedBlockBandedMatrix{T}},n::Integer,m::Integer) where {T<:Number}
    resizedata!(B, :, m)
    if n < B.datasize[1]
        return B
    end

    l,u,λ,μ = B.data.l,B.data.u,B.data.λ,B.data.μ

    l,u,λ,μ = B.data.l,B.data.u,B.data.λ,B.data.μ

    # make sure we have enough rows
    K = Int(block(rangespace(B),n))
    rows = blocklengths(rangespace(B.op))[1:K]

    B.data = _BandedBlockBandedMatrix(B.data.data, blockedrange(rows), (l,u), (λ,μ))
    B.datasize = (n,B.datasize[2])

    B
end
