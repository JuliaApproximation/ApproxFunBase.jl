mutable struct RaggedMatrix{T} <: AbstractMatrix{T}
    const data::Vector{T} # a Vector of non-zero entries
    const cols::Vector{Int} # a Vector specifying the first index of each column
    m::Int #Number of rows
    function RaggedMatrix{T}(data::Vector{T}, cols::Vector{Int}, m::Int) where T
        ragged_checks(data, cols, m)
        new{T}(data,cols,m)
    end
end
