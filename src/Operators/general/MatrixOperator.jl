function Conversion(A::Array{<:Any,1},B::Array{<:Any,1},plan::Array{Int64,1})
    @assert size(A)==size(B) && size(A)==size(plan) # lazy assertion
    ret=Any[ZeroOperator(promote_type(prectype(AA),prectype(BB)),AA,BB) for BB in B, AA in A]
    for k in 1:length(A)
        ret[plan[k],k]=Conversion(A[k],B[plan[k]])
    end
    ret
end
