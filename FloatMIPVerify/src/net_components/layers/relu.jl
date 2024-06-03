export ReLU

"""
$(TYPEDEF)

Represents a ReLU operation.

`p(x)` is shorthand for [`relu(x)`](@ref) when `p` is an instance of
`ReLU`.
"""
struct ReLU <: Layer
    tightening_algorithm::Union{TighteningAlgorithm,Nothing}
    model_pyobj::Union{PyObject,Nothing}
end

#不同的外部函数warper来实现不同的构造结构体对象的需求
function ReLU()::ReLU
    ReLU(nothing, nothing)
end

# function ReLU(algorithm::TighteningAlgorithm, model_pyobj::PyObject)
#     ReLU(algorithm, model_pyobj)
# end

function ReLU(algorithm::TighteningAlgorithm)
    ReLU(algorithm, nothing)
end

function ReLU(model_pyobj::PyObject)
    ReLU(nothing,model_pyobj)
end



function Base.show(io::IO, p::ReLU)
    print(io, "ReLU()")
end

(p::ReLU)(x::Array{<:Real}) = relu(x)
(p::ReLU)(x::Array{<:JuMPLinearType}) =
    (Memento.info(MIPVerify.LOGGER, "Applying $p ..."); relu(x, nta = p.tightening_algorithm))
(p::ReLU)(x::Array{<:JuMPLinearType}, x_itv::Array{<:PyObject}) =
    (Memento.info(MIPVerify.LOGGER, "Applying $p ..."); relu(p.model_pyobj, x, x_itv, nta = p.tightening_algorithm))



