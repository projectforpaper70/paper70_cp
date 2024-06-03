export SequentialItv 

"""
$(TYPEDEF)

Represents a sequential (feed-forward) neural net, with `layers` ordered from input
to output.

## Fields:
$(FIELDS)
"""
struct SequentialItv <: NeuralNet
    layers::Array{Layer,1}
    UUID::String
end 

function Base.show(io::IO, p::SequentialItv)

    println(io, "sequential net $(p.UUID)")
    for (index, value) in enumerate(p.layers)
        println(io, "  ($index) $value")
    end
end


function apply(p::SequentialItv, x::Array{<:JuMPReal},x_itv::Array{<:PyObject})
    input1,input2 = x,x_itv
    linear_layer_count = 1
    relu_layer_count = 1
    for layer in p.layers
        if typeof(layer) <: Two_Param_Linear
            #println("当前linear: ", linear_layer_count)
            linear_layer_count += 1
            output1,output2 = layer(input1,input2)
        elseif typeof(layer) <: ReLU
            #println("当前relu: ", relu_layer_count)
            relu_layer_count += 1
            output1,output2 = layer(input1,input2)
        end
        input1,input2 = output1,output2
    end
    return input1,input2
end


(p::SequentialItv)(x::Array{<:JuMPReal},x_itv::Array{<:PyObject}) = apply(p, x, x_itv)
