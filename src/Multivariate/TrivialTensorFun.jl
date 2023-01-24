


struct TrivialTensorFun{d, SS<:TensorSpaceND{d}, T<:Number} <: MultivariateFun{T, d}
    space::SS
    coefficients::Vector{T}
    iterator::TrivialTensorizer{d}
    orders::Block{1, Int}
end


function TrivialTensorFun(iter::TrivialTensorizer{d},cfs::Vector{T},blk::Block, sp::TensorSpaceND{d}) where {T<:Number,d}
    if any(map(dimension, sp.spaces).!=ℵ₀)
        error("This Space is not a Trivial Tensor space!")
    end
    TrivialTensorFun(sp, cfs, iter, blk)
end

(f::TrivialTensorFun)(x...) = evaluate(f, x...)

# TensorSpace evaluation
function evaluate(f::TrivialTensorFun{d, SS, T},x...) where {d, SS, T}
    highest_order = f.orders.n[1]
    n = length(f.coefficients)

    # this could be lazy evaluated for the sparse case
    A = T[Fun(f.space.spaces[i], [zeros(T, k);1])(x[i]) for k=0:highest_order, i=1:d]
    result::T = 0
    coef_counter::Int = 1
    for i in f.iterator
        tmp = f.coefficients[coef_counter]
        if tmp != 0
            tmp_res = 1
            for k=1:d
                tmp_res *= A[i[k], k]
            end
            result += tmp * tmp_res
        end
        coef_counter += 1
        if coef_counter > n
            break
        end
    end
    return result
end
