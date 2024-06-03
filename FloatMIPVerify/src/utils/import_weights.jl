export get_matrix_params, get_conv_params, get_example_network_params
#include("paramInterval.jl")

"""
$(SIGNATURES)

Helper function to import the parameters for a layer carrying out matrix multiplication
    (e.g. fully connected layer / softmax layer) from `param_dict` as a
    [`Linear`](@ref) object.

The default format for the key is `'layer_name/weight'` and `'layer_name/bias'`;
    you can customize this by passing in the named arguments `matrix_name` and `bias_name`
    respectively. The expected parameter names will then be `'layer_name/matrix_name'`
    and `'layer_name/bias_name'`

# Arguments
* `param_dict::Dict{String}`: Dictionary mapping parameter names to array of weights
    / biases.
* `layer_name::String`: Identifies parameter in dictionary.
* `expected_size::NTuple{2, Int}`: Tuple of length 2 corresponding to the expected size
   of the weights of the layer.

"""
function get_matrix_params(
    param_dict::Dict{String, Any},
    layer_name::String,
    expected_size::NTuple{2, Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    data_type::Type = Float32,
    trans2itv::Bool =false,
    debug_test::Bool = false,
    to_double::Bool = false
)::Layer

    param_weight = param_dict["$layer_name/$matrix_name"]
    param_bias = dropdims(param_dict["$layer_name/$bias_name"], dims=1)


    if trans2itv
        #如果需要将参数16位浮点化
        if data_type == Float16
            param_weight = Float16.(param_weight)
            param_bias = Float16.(param_bias)
            #如果需要将参数16位区间化，并使用相关的线性层包裹
            if trans2itv
                param_weight_itv = weight2interval16(param_weight)
                param_bias_itv = bias2interval16(param_bias)
                param = Two_Param_Linear(param_weight,param_bias,param_weight_itv,param_bias_itv)
                return param
            end
        elseif data_type ==Float32
            param_weight = Float32.(param_weight)
            param_bias = Float32.(param_bias)
            #如果需要将参数16位区间化，并使用相关的线性层包裹
            if trans2itv
                param_weight_itv = weight2interval32(param_weight)
                param_bias_itv = bias2interval32(param_bias)
                param = Two_Param_Linear(param_weight,param_bias,param_weight_itv,param_bias_itv)
                return param
            end
        end
    else
        #如果需要将参数16位浮点化
        if data_type == Float16
            param_weight = Float16.(param_weight)
            param_bias = Float16.(param_bias)
        elseif data_type ==Float32
            param_weight = Float32.(param_weight)
            param_bias = Float32.(param_bias)
        end

    end

    if(debug_test)
        println("type of weight is :",typeof(param_weight))
        #不需要额外操作直接返回
        println(param_bias)

        println("\n")
        println(param_weight)

    
    end

    if(to_double)
        param_weight = Float64.(param_weight)
        param_bias = Float64.(param_bias)
    end
    params = Linear(param_weight, param_bias)
    check_size(params, expected_size)
    
    return params
end




"""
Helper function to import the nnet mnist24 models

# Arguments



"""
function get_matrix_params_nnet(
    nnet_model::NNetFormat,
    layer_index::Int,
    expected_size::NTuple{2,Int};
)::Linear

    params = Linear(
        Matrix(transpose(nnet_model.weights[layer_index])),
        nnet_model.biases[layer_index],
    )

    check_size(params,expected_size)

    return params
    
end

"""
$(SIGNATURES)

Helper function to import the parameters for a convolution layer from `param_dict` as a
    [`Conv2d`](@ref) object.

The default format for the key is `'layer_name/weight'` and `'layer_name/bias'`;
    you can customize this by passing in the named arguments `matrix_name` and `bias_name`
    respectively. The expected parameter names will then be `'layer_name/matrix_name'`
    and `'layer_name/bias_name'`

# Arguments
* `param_dict::Dict{String}`: Dictionary mapping parameter names to array of weights
    / biases.
* `layer_name::String`: Identifies parameter in dictionary.
* `expected_size::NTuple{4, Int}`: Tuple of length 4 corresponding to the expected size
    of the weights of the layer.

"""
function get_conv_params(
    param_dict::Dict{String},
    layer_name::String,
    expected_size::NTuple{4,Int};
    matrix_name::String = "weight",
    bias_name::String = "bias",
    expected_stride::Integer = 1,
    padding::Padding = SamePadding(),
)::Conv2d

    params = Conv2d(
        param_dict["$layer_name/$matrix_name"],
        dropdims(param_dict["$layer_name/$bias_name"], dims = 1),
        expected_stride,
        padding,
    )

    check_size(params, expected_size)

    return params
end
