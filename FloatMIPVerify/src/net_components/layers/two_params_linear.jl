export Two_Param_Linear

"""
$(TYPEDEF)

Represents matrix multiplication.

`p(x)` is shorthand for [`matmul(x, p)`](@ref) when `p` is an instance of
`Itv_Linear`.

## Fields:
$(FIELDS)
"""
# #这个是为了适配nnet网络格式的版本
# struct Two_Param_Linear{T1<:Real, U1<:Real, T2<:Interval, U2<:Interval} <: Layer
#     matrix1::Array{T1,2}
#     bias1::Array{U1,1}
#     matrix2::Array{T2,2}
#     bias2::Array{U2,1}

#     function Two_Param_Linear{T1,U1,T2,U2}(matrix1::Array{T1,2}, bias1::Array{U1,1},  matrix2::Array{T2,2}, bias2::Array{U2,1}) where {T1<:Real, U1<:Real, T2<:Interval,U2<:Interval}
#         (matrix_width1, matrix_height1) = size(matrix1)
#         (matrix_width2, matrix_height2) = size(matrix2)
#         bias_height1 = length(bias1)
#         bias_height2 = length(bias2)
#         @assert(
#             matrix_width1 == bias_height1,
#             "Number of output channels in matrix, $matrix_width1, does not match number of output channels in bias, $bias_height1."
#         )
#         @assert(
#             matrix_width2 == bias_height2,
#             "Number of output channels in matrix, $matrix_width2, does not match number of output channels in bias, $bias_height2."
#         )
#         return new(matrix1, bias1, matrix2, bias2)
#     end

# end

struct Two_Param_Linear{T1<:Real, U1<:Real, T2<:Interval, U2<:Interval} <: Layer
    matrix1::Array{T1,2}
    bias1::Array{U1,1}
    matrix2::Array{T2,2}
    bias2::Array{U2,1}

    function Two_Param_Linear{T1,U1,T2,U2}(matrix1::Array{T1,2}, bias1::Array{U1,1},  matrix2::Array{T2,2}, bias2::Array{U2,1}) where {T1<:Real, U1<:Real, T2<:Interval,U2<:Interval}
        (matrix_width1, matrix_height1) = size(matrix1)
        (matrix_width2, matrix_height2) = size(matrix2)
        bias_height1 = length(bias1)
        bias_height2 = length(bias2)
        @assert(
            matrix_height1 == bias_height1,
            "Number of output channels in matrix, $matrix_height1, does not match number of output channels in bias, $bias_height1."
        )
        @assert(
            matrix_height2 == bias_height2,
            "Number of output channels in matrix, $matrix_height2, does not match number of output channels in bias, $bias_height2."
        )
        return new(matrix1, bias1, matrix2, bias2)
    end

end

function Two_Param_Linear(matrix1::Array{T1,2}, bias1::Array{U1,1},  matrix2::Array{T2,2}, bias2::Array{U2,1}) where {T1<:Real, U1<:Real, T2<:Interval,U2<:Interval}
    Two_Param_Linear{T1,U1,T2,U2}(matrix1, bias1, matrix2, bias2)
end


function matmul(x::Array{T1,1}, x_itv::Array{T2,1}, params::Two_Param_Linear{U1,V1,U2,V2}) where {T1<:JuMPLinearType, T2<:PyObject, U1<:Real, V1<:Real, U2<:Interval,V2<:Interval}

    #jump part
    Memento.info(MIPVerify.LOGGER, "Applying $params ... ")
    #println("Abstract matmul the size of input is :")
    #println(size(x))
    (matrix_height1, matrix_width1) = size(params.matrix1)
    (input_height1,) = size(x)
    @assert(
        matrix_height1 == input_height1,
        "Number of values in input, $input_height1, does not match number of values, $matrix_height1 that Linear operates on."
    )

    #PyObject part
    Weight=transpose(params.matrix2)
    (matrix_height2, matrix_width2) = size(Weight)
    (input_height2,) = size(x_itv)
    @assert(
        matrix_width2 == input_height2,
        "Number of values in input, $input_height2, does not match number of values, $matrix_height2 that Linear operates on."
    )
    pre_var = PyObject[]
    for i in 1:matrix_height2
        var_tmp_l=sum(Weight[i,j].lo*x_itv[j] for j in 1:matrix_width2)+params.bias2[i].lo
        var_tmp_u=sum(Weight[i,j].hi*x_itv[j] for j in 1:matrix_width2)+params.bias2[i].hi
        #sum(var_tmp,Bias[i])
        var_tmp = Interval_Var(var_tmp_u,var_tmp_l)
        push!(pre_var,var_tmp) 
    end
    #return pre_var

    return transpose(params.matrix1) * x .+ params.bias1, pre_var
end

# #这个版本的是nnet类型网络的乘法操作，不需要再对权重进行转置操作
# function matmul(x::Array{T1,1}, x_itv::Array{T2,1}, params::Two_Param_Linear{U1,V1,U2,V2}) where {T1<:JuMPLinearType, T2<:PyObject, U1<:Real, V1<:Real, U2<:Interval,V2<:Interval}

#     #jump part
#     Memento.info(MIPVerify.LOGGER, "Applying $params ... ")
#     #println("Abstract matmul the size of input is :")
#     #println(size(x))
#     (matrix_height1, matrix_width1) = size(params.matrix1)
#     (input_height1,) = size(x)
#     @assert(
#         matrix_width1 == input_height1,
#         "Number of values in input, $matrix_width1, does not match number of values, $matrix_height1 that Linear operates on."
#     )

#     #PyObject part
#     Weight=params.matrix2
#     (matrix_height2, matrix_width2) = size(Weight)
#     (input_height2,) = size(x_itv)
#     @assert(
#          matrix_width2 == input_height2,
#         "Number of values in input, $matrix_width2, does not match number of values, $matrix_height2 that Linear operates on."
#     )
#     pre_var = PyObject[]
#     for i in 1:matrix_height2
#         # println("type of weights elems and x_itv elems are: \n")
#         # println(typeof(Weight[i,1].lo))
#         # println("\n")
#         # println(typeof(x_itv[1]))
#         var_tmp_l=sum(Weight[i,j].lo*x_itv[j] for j in 1:matrix_width2)+params.bias2[i].lo
#         var_tmp_u=sum(Weight[i,j].hi*x_itv[j] for j in 1:matrix_width2)+params.bias2[i].hi
#         #sum(var_tmp,Bias[i])
#         var_tmp = Interval_Var(var_tmp_u,var_tmp_l)
#         push!(pre_var,var_tmp) 
#     end
#     #return pre_var

#     return params.matrix1 * x .+ params.bias1, pre_var
# end


(p::Two_Param_Linear)(x::Array{<:JuMPReal},x_itv::Array{<:PyObject}) =
    "Linear() layers work only on one-dimensional input. You likely forgot to add a Flatten() layer before your first linear layer." |>
    ArgumentError |>
    throw

(p::Two_Param_Linear)(x::Array{<:JuMPReal,1},x_itv::Array{<:PyObject,1}) = matmul(x,x_itv, p)


