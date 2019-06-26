function Conversion(A::Array{Space,1},B::Array{Space,1},plan::Array{Int64,1})
    @assert size(A)==size(B) && size(A)==size(plan) # lazy assertion
    ret=[ZeroOperator(AA,BB) for BB in B, AA in A]
    for k in 1:length(A)
        ret[plan[k],k]=Conversion(A[k],B[k])
    end
    ret
end
