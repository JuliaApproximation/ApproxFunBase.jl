module ApproxFunBaseSparseArraysExt

using SparseArrays
import SparseArrays: blockdiag
using ApproxFunBase
using ApproxFunBase: promote_eltypeof

##TODO: unify with other blockdiag
function blockdiag(d1::AbstractVector{T}, d2::AbstractVector{T}) where T<:Operator
    if isempty(d1) && isempty(d2)
        error("Empty blockdiag")
    end
    if isempty(d1)
        TT=promote_eltypeof(d2)
    elseif isempty(d2)
        TT=promote_eltypeof(d1)
    else
        TT=promote_type(promote_eltypeof(d1),
                        promote_eltypeof(d2))
    end

      D=zeros(Operator{TT},length(d1)+length(d2),2)
      D[1:length(d1),1]=d1
      D[length(d1)+1:end,2]=d2
      D
end

function blockdiag(a::Operator, b::Operator)
	blockdiag(Operator{promote_type(eltype(a),eltype(b))}[a],
			Operator{promote_type(eltype(a),eltype(b))}[b])
end

blockdiag(A::PlusOperator) = mapreduce(blockdiag, +, A.ops)
blockdiag(A::TimesOperator) = mapreduce(blockdiag, .*, A.ops)

end
