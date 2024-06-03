export Itv_Linear

"""
$(TYPEDEF)

Represents matrix multiplication.

`p(x)` is shorthand for [`matmul(x, p)`](@ref) when `p` is an instance of
`Itv_Linear`.

## Fields:
$(FIELDS)
"""
# #这里的itv_Linear是为了适配nnet模型所做的修改版本
# struct Itv_Linear{T<:Interval,U<:Interval} <: Layer
#     matrix::Array{T,2}
#     bias::Array{U,1}

#     function Itv_Linear{T,U}(matrix::Array{T,2}, bias::Array{U,1}) where {T<:Interval,U<:Interval}
#         (matrix_width, matrix_height) = size(matrix)
#         bias_height = length(bias)
#         @assert(
#             matrix_width == bias_height,
#             "Number of output channels in matrix, $matrix_width, does not match number of output channels in bias, $bias_height."
#         )
#         return new(matrix, bias)
#     end

# end

struct Itv_Linear{T<:Interval,U<:Interval} <: Layer
    matrix::Array{T,2}
    bias::Array{U,1}

    function Itv_Linear{T,U}(matrix::Array{T,2}, bias::Array{U,1}) where {T<:Interval,U<:Interval}
        (matrix_width, matrix_height) = size(matrix)
        bias_height = length(bias)
        @assert(
            matrix_height == bias_height,
            "Number of output channels in matrix, $matrix_height, does not match number of output channels in bias, $bias_height."
        )
        return new(matrix, bias)
    end

end

function Itv_Linear(matrix::Array{T,2}, bias::Array{U,1}) where {T<:Interval,U<:Interval}
    Itv_Linear{T,U}(matrix, bias)
end

function Base.show(io::IO, p::Itv_Linear)
    input_size = size(p.matrix)[1]
    output_size = size(p.matrix)[2]
    print(io, "Itv_Linear($input_size -> $output_size)")
end

function check_size(params::Itv_Linear, sizes::NTuple{2,Int})::Nothing
    check_size(params.matrix, sizes)
    check_size(params.bias, (sizes[end],))
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`.
"""
function matmul(x::Array{<:Interval,1}, params::Itv_Linear)
    return transpose(params.matrix) * x .+ params.bias
end

"""
$(SIGNATURES)

Computes the result of pre-multiplying `x` by the transpose of `params.matrix` and adding
`params.bias`. We write the computation out by hand when working with `JuMPItv_LinearType`
so that we are able to simplify the output as the computation is carried out.
"""

#抽象矩阵乘法操作
function matmul(x::Array{T,1},params::Itv_Linear{U,V}) where {T<:PyObject,U<:Interval,V<:Interval}
    Weight=transpose(params.matrix)
    (matrix_height, matrix_width) = size(Weight)
    (input_height,) = size(x)
    @assert(
        matrix_width == input_height,
        "Number of values in input, $input_height, does not match number of values, $matrix_height that Linear operates on."
    )
    pre_var = PyObject[]
    for i in 1:matrix_height
        var_tmp_l=sum(Weight[i,j].lo*x[j] for j in 1:matrix_width)+params.bias[i].lo
        var_tmp_u=sum(Weight[i,j].hi*x[j] for j in 1:matrix_width)+params.bias[i].hi
        #sum(var_tmp,Bias[i])
        var_tmp = Interval_Var(var_tmp_u,var_tmp_l)
        push!(pre_var,var_tmp)
    end
    return pre_var
end

(p::Itv_Linear)(x::Array{<:PyObReal}) =
    "Linear() layers work only on one-dimensional input. You likely forgot to add a Flatten() layer before your first linear layer." |>
    ArgumentError |>
    throw

#(p::Linear)(x::Array{<:JuMPReal,1}) = matmul(x, p)

(p::Itv_Linear)(x::Array{<:PyObReal,1}) = matmul(x, p)



