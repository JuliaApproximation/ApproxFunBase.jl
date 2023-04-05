function CachedOperator(::Type{BandedMatrix},op::Operator;padding::Bool=false)
    bw = bandwidths(op)
    l = first(bw)
    # working on the tuples directly instead of the components helps with type-stability
    padding && (bw = bw .+ (0,l))
    data = BandedMatrix{eltype(op)}(undef, (0,0), bw)
    CachedOperator(op,data,size(data),domainspace(op),rangespace(op), bw .* (-1,1),padding)
end



## Grow cached operator

function resizedata!(B::CachedOperator{T,<:BandedMatrix{T}},n::Integer,m_in::Integer) where T<:Number
    m = max(m_in,n+B.data.u)
    N,M = size(B)
    n = min(n, N)

    if n > B.datasize[1]
        pad!(B.data,min(N,2n),m)

        kr=B.datasize[1]+1:n
        jr=max(B.datasize[1]+1-B.data.l,1):min(n+B.data.u,M)
        axpy!(1.0,view(B.op,kr,jr),view(B.data,kr,jr))

        B.datasize = (n,m)
    end

    B
end


resizedata!(B::CachedOperator{T,<:BandedMatrix{T}},n::Integer,::Colon) where T<:Number =
    resizedata!(B, n, n+B.data.u)

## Grow QR

function QROperator(R::CachedOperator{T,<:BandedMatrix{T}}) where T
    M = R.data.l+1   # number of diag+subdiagonal bands
    H = Array{T}(undef,M,100)
    QROperator(R,H,0)
end


function resizedata!(QR::QROperator{<:CachedOperator{T,<:BandedMatrix{T}}}, ::Colon,col) where {T}
    if col ≤ QR.ncols
        return QR
    end

    col = min(col, size(QR,2))

    MO=QR.R_cache
    W=QR.H

    R=MO.data
    M=R.l+1   # number of diag+subdiagonal bands

    if col+M-1 ≥ MO.datasize[1]
        resizedata!(MO,(col+M-1)+100,:)  # double the last rows
    end

    if col > size(W,2)
        W=QR.H=unsafe_resize!(W,:,2col)
    end

    for k=QR.ncols+1:col
        W[:,k] = view(R.data,R.u+1:R.u+R.l+1,k) # diagonal and below
        wp=view(W,:,k)
        W[1,k]+= flipsign(norm(wp),W[1,k])
        normalize!(wp)

        # scale banded entries
        for j=k:k+R.u
            dind=R.u+1+k-j
            v=view(R.data,dind:dind+M-1,j)
            dt=dot(wp,v)
            axpy!(-2dt,wp,v)
        end

        # scale banded/filled entries
        for j=k+R.u+1:k+R.u+M-1
            p=j-k-R.u
            v=view(R.data,1:M-p,j)  # shift down each time
            wp2=view(wp,p+1:M)
            dt=dot(wp2,v)
            axpy!(-2dt,wp2,v)
        end
    end
    QR.ncols=col
    QR
end


## back substitution
# loop to avoid ambiguity with AbstractTRiangular
for ArrTyp in (:AbstractVector, :AbstractMatrix, :StridedVector)
    @eval function ldiv!(U::UpperTriangular{T, <:SubArray{T, 2, <:BandedMatrix{T}, NTuple{2,UnitRange{Int}}, false}},
                             u::$ArrTyp{T}) where T
        n = size(u,1)
        n == size(U,1) || throw(DimensionMismatch())

        V = parent(U)
        @assert first(parentindices(V)[1]) == 1
        @assert first(parentindices(V)[2]) == 1

        A = parent(V)

        b=bandwidth(A,2)

        for c=1:size(u,2)
            for k=n:-1:1
                @simd for j=k+1:min(n,k+b)
                    @inbounds u[k,c] = muladd(-A.data[k-j+A.u+1,j],u[j,c],u[k,c])
                end

                @inbounds u[k,c] /= A.data[A.u+1,k]
            end
        end
        u
    end
end
