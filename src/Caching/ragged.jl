
CachedOperator(::Type{RaggedMatrix},op::Operator;padding::Bool=false) =
    CachedOperator(op,RaggedMatrix{eltype(op)}(undef, 0, Int[]),padding)

## Grow cached operator

function resizedata!(B::CachedOperator{T,RaggedMatrix{T}},::Colon,n::Integer) where T<:Number
    if n > size(B,2)
        throw(ArgumentError("Cannot resize beyond size of operator"))
    end

    if n > B.datasize[2]
        resize!(B.data.cols,n+1)

        if B.padding
            # K is largest colstop.  We get previous largest by looking at precalulated
            # cols
            K = B.datasize[2]==0 ? 0 : B.data.cols[B.datasize[2]+1]-B.data.cols[B.datasize[2]]

            for j = B.datasize[2]+1:n
                K = max(K,colstop(B.op,j))
                B.data.cols[j+1] = B.data.cols[j] + K
            end
        else
            K = B.datasize[2]==0 ? 0 : B.data.m# more robust but slower: maximum(diff(B.data.cols))

            for j = B.datasize[2]+1:n
                cs = colstop(B.op,j)
                K = max(K,cs)
                B.data.cols[j+1] = B.data.cols[j] + cs
            end
        end

        # avoid padding with negative length
        if B.data.cols[n+1] ≤ 0
            return B
        end

        pad!(B.data.data,B.data.cols[n+1]-1)
        B.data = RaggedMatrix(B.data.data, B.data.cols, K)

        jr=B.datasize[2]+1:n
        kr=1:K
        axpy!(1.0,view(B.op,kr,jr),view(B.data,kr,jr))

        B.datasize = (K,n)

    end

    B
end

function resizedata!(B::CachedOperator{T,RaggedMatrix{T}},n::Integer,m::Integer) where T<:Number
    resizedata!(B,:,m)
    data = B.data
    B.data = RaggedMatrix(data.data, data.cols, max(data.m,n))   # make sure we have at least n rows
    B
end


## Grow QR

QROperator(R::CachedOperator{T,RaggedMatrix{T}}) where {T} =
    QROperator(R,RaggedMatrix{T}(undef,0,Int[]),0)

function resizedata!(QR::QROperator{<:CachedOperator{T,RaggedMatrix{T}}, <:RaggedMatrix}, ::Colon, col) where {T}
    if col ≤ QR.ncols
        return QR
    end

    MO=QR.R_cache
    W=QR.H

    if col > MO.datasize[2]
        m = MO.datasize[2]
        resizedata!(MO,:,col+100)  # last rows plus a bunch more

        # apply previous Householders to new columns of R
        for J=1:QR.ncols
            wp=view(W,1:colstop(W,J),J)
            for j = m+1:MO.datasize[2]
                kr = range(J, length=length(wp))
                v = view(MO.data,kr,j)
                dt = dot(wp,v)
                axpy!(-2dt, wp, v)
            end
        end
    end


    if col > size(W,2)
        m = size(W,2)
        resize!(W.cols,col+101)

        Wm = W.m
        for j=m+1:col+100
            cs = colstop(MO.data,j)
            W.cols[j+1] = W.cols[j] + cs-j+1
            Wm = max(Wm,cs-j+1)
        end

        resize!(W.data,W.cols[end]-1)
        W = RaggedMatrix(W.data, W.cols, Wm)
        QR.H = W
    end

    for k=QR.ncols+1:col
        cs = colstop(MO.data,k)
        indsk = k:cs
        indskax = eachindex(indsk)
        W[indskax,k] = view(MO.data,indsk,k) # diagonal and below
        wp = view(W,indskax,k)
        W[1,k] += flipsign(norm(wp),W[1,k])
        normalize!(wp)

        # scale rows entries
        kr = range(k, length=length(wp))
        for j=k:MO.datasize[2]
            v = view(MO.data,kr,j)
            dt = dot(wp,v)
            axpy!(-2dt, wp, v)
        end
    end
    QR.ncols=col
    QR
end


## back substitution
for ArrTyp in (:AbstractVector, :AbstractMatrix)
    @eval function ldiv!(U::UpperTriangular{T, <:SubArray{T, 2, RaggedMatrix{T}, NTuple{2,UnitRange{Int}}}},
                             u::$ArrTyp{T}) where T
        n = size(u,1)
        n == size(U,1) || throw(DimensionMismatch())

        V = parent(U)
        @assert parentindices(V)[1][1] == 1
        @assert parentindices(V)[2][1] == 1

        A = parent(V)

        for c=axes(u,2)
            for k=reverse(axes(u,1))
                ck = A.cols[k]
                u[k,c] /= A.data[ck+k-1]
                axpy!(-u[k,c], view(A.data, range(ck, length=k-1)), view(u,1:k-1,c))
            end
        end
        u
    end
end



## Apply Q


function mulpars(Ac::Adjoint{T,<:QROperatorQ{QROperator{RR,RaggedMatrix{T},T},T}},
                      B::AbstractVector{T},tolerance,maxlength) where {RR,T}
    A = parent(Ac)
    if length(B) > A.QR.ncols
        # upper triangularize extra columns to prepare for \
        resizedata!(A.QR,:,length(B)+size(A.QR.H,1)+10)
    end

    H=A.QR.H
    M=size(H,1)
    m=length(B)
    Y=pad(B,m+M+10)

    k=1
    yp=view(Y,1:length(B))
    while (k ≤ m || norm(yp) > tolerance )
        if k > maxlength
            @warn "Maximum length $maxlength reached."
            break
        end
        if k > A.QR.ncols
            # upper triangularize extra columns to prepare for \
            resizedata!(A.QR,:,k+M+50)
            H=A.QR.H
            M=size(H,1)
        end

        cr=colrange(H,k)

        if k+length(cr)-1>length(Y)
            pad!(Y,2*(k+M))
        end

        wp=view(H,cr,k)
        yp=view(Y, (k-1) .+ cr)

        dt=dot(wp,yp)
        axpy!(-2dt,wp,yp)
        k+=1
    end
    nz = findlast(!iszero, Y)
    resize!(Y,nz === nothing ? k : min(k, nz))  # chop off zeros
end
