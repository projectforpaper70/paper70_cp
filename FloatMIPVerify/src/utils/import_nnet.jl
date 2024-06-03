export NNetFormat, evaluate_network, evaluate_network_multiple, num_inputs, num_outputs
mutable struct NNetFormat
    weights::Array{Any,1}
    biases::Array{Any,1}
    numLayers::Int32
    layerSizes::Array{Int32,1}
    inputSize::Int32
    outputSize::Int32
    mins::Array{Float64,1}
    maxes::Array{Float64,1}
    means::Array{Float64,1}
    ranges::Array{Float64,1}
    
    function NNetFormat(file::AbstractString,modeltype::String)
        this  = new()
        
        # Open file for reading
        f = open(file)
        
        # Skip any header lines
        line = readline(f)
        while line[1:2]=="//"
            line = readline(f)
        end
        
        # Read information about the neural network
        record = split(line,[',','\n'])
        this.numLayers = parse(Int32,record[1])
        this.inputSize = parse(Int32,record[2])
        this.outputSize = parse(Int32,record[3])
        
        line = readline(f)
        record = split(line,[',','\n'])
        this.layerSizes = zeros(this.numLayers+1)
        for i=1:(this.numLayers+1)
            this.layerSizes[i]=parse(Int32,record[i])
        end
        
        # Skip unused line
        line = readline(f)

        if(modeltype == "ACASXU")
            line = readline(f)
            record = split(line,[',','\n'])
            this.mins = zeros(this.inputSize)
            for i=1:(this.inputSize)
                this.mins[i]=parse(Float64,record[i])
            end
            
            line = readline(f)
            record = split(line,[',','\n'])
            this.maxes = zeros(this.inputSize)
            for i=1:(this.inputSize)
                this.maxes[i]=parse(Float64,record[i])
            end
            
            line = readline(f)
            record = split(line,[',','\n'])
            this.means = zeros(this.inputSize+1)
            for i=1:(this.inputSize+1)
                this.means[i]=parse(Float64,record[i])
            end
            
            line = readline(f)
            record = split(line,[',','\n'])
            this.ranges = zeros(this.inputSize+1)
            for i=1:(this.inputSize+1)
                this.ranges[i]=parse(Float64,record[i])
            end
        end
        
        
        # Initialize weight and bias arrays
        this.weights = Any[zeros(this.layerSizes[2],this.layerSizes[1])]
        this.biases  = Any[zeros(this.layerSizes[2])]
        for i=2:this.numLayers
            this.weights = [this.weights;Any[zeros(this.layerSizes[i+1],this.layerSizes[i])]]
            this.biases  = [this.biases;Any[zeros(this.layerSizes[i+1])]]
        end

        # Fill weight and bias arrays with values from nnet file
        layer=1
        i=1
        j=1
        line = readline(f)
        record = split(line,[',','\n'])
        while !eof(f)
            while i<=this.layerSizes[layer+1]
                while record[j]!=""
                    this.weights[layer][i,j] = parse(Float64,record[j])
                    j=j+1
                end
                j=1
                i=i+1
                line = readline(f)
                record = split(line,[',','\n'])
            end
            i=1
            while i<=this.layerSizes[layer+1]
                this.biases[layer][i] = parse(Float64,record[1])
                i=i+1
                line = readline(f)
                record = split(line,[',','\n'])
            end
            layer=layer+1
            i=1
            j=1
        end
        close(f)
        
        return this
    end
end

"""
Evaluate network using given inputs

Args:
    nnet (NNet): Neural network to evaluate
    inputs (array): Network inputs to be evaluated

Returns:
    (array): Network output
"""
function evaluate_network(nnet::NNetFormat,input::Array{Float64,1})
    numLayers = nnet.numLayers
    inputSize = nnet.inputSize
    outputSize = nnet.outputSize
    biases = nnet.biases
    weights = nnet.weights
    
    # Prepare the inputs to the neural network
    inputs = zeros(inputSize)
    for i = 1:inputSize
        if input[i]<nnet.mins[i]
            inputs[i] = (nnet.mins[i]-nnet.means[i])/nnet.ranges[i]
        elseif input[i] > nnet.maxes[i]
            inputs[i] = (nnet.maxes[i]-nnet.means[i])/nnet.ranges[i] 
        else
            inputs[i] = (input[i]-nnet.means[i])/nnet.ranges[i] 
        end
    end

    # Evaluate the neural network
    for layer = 1:numLayers-1
        temp = max.(*(weights[layer],inputs[1:nnet.layerSizes[layer]])+biases[layer],0)
        inputs = temp
    end
    outputs = *(weights[end],inputs[1:nnet.layerSizes[end-1]])+biases[end]
    
    # Undo output normalization
    for i=1:outputSize
        outputs[i] = outputs[i]*nnet.ranges[end]+nnet.means[end]
    end
    return outputs
end


function evaluate_network_withoutNorm(nnet::NNetFormat, input::Array{Float64, 1})
    numLayers = nnet.numLayers
    biases = nnet.biases
    weights = nnet.weights

    # 评估神经网络
    inputs = input
    for layer = 1:numLayers-1
        temp = max.(weights[layer] * inputs + biases[layer], 0)
        inputs = temp
    end
    outputs = weights[end] * inputs + biases[end]

    return outputs
end

"""
Evaluate network using multiple sets of inputs

Args:
    nnet (NNet): Neural network to evaluate
    inputs (array): Network inputs to be evaluated

Returns:
    (array): Network outputs for each set of inputs
"""
function evaluate_network_multiple(nnet::NNetFormat,input::Array{Float64,2})
    numLayers = nnet.numLayers
    inputSize = nnet.inputSize
    outputSize = nnet.outputSize
    biases = nnet.biases
    weights = nnet.weights
     
    # Prepare the inputs to the neural network
    _,numInputs = size(input)
    inputs = zeros(inputSize,numInputs)
    for i = 1:inputSize
        for j = 1:numInputs
            if input[i,j]<nnet.mins[i]
                inputs[i,j] = (nnet.mins[i]-nnet.means[i])/nnet.ranges[i]
            elseif input[i,j] > nnet.maxes[i]
                inputs[i,j] = (nnet.maxes[i]-nnet.means[i])/nnet.ranges[i] 
            else
                inputs[i,j] = (input[i,j]-nnet.means[i])/nnet.ranges[i] 
            end
        end
    end

    # Evaluate the neural network
    for layer = 1:numLayers-1
        inputs = max.(*(weights[layer],inputs[1:nnet.layerSizes[layer],:])+*(biases[layer],ones(1,numInputs)),0)
    end
    outputs = *(weights[end],inputs[1:nnet.layerSizes[end-1],:])+*(biases[end],ones(1,numInputs))
    
    # Undo output normalization
    for i=1:outputSize
        for j=1:numInputs
            outputs[i,j] = outputs[i,j]*nnet.ranges[end]+nnet.means[end]
        end
    end
    return outputs
end

""" Get number of inputs to network"""
function num_inputs(nnet::NNetFormat)
    return nnet.inputSize
end

""" Get number of outputs from network"""
function num_outputs(nnet::NNetFormat)
    return nnet.outputSize
end


"""打印NNet对象的信息"""
function info(nnet::NNetFormat)
    println("神经网络的层数：", nnet.numLayers)
    println("各层神经元数量的数组：", nnet.layerSizes)
    println("神经网络的输入数量：", nnet.inputSize)
    println("神经网络的输出数量：", nnet.outputSize)
    println("输入特征的最小值数组：", nnet.mins)
    println("输入特征的最大值数组：", nnet.maxes)
    println("输入特征的均值数组：", nnet.means)
    println("输入特征的范围数组：", nnet.ranges)
end

# weights::Array{Any,1}
# biases::Array{Any,1}
# numLayers::Int32
# layerSizes::Array{Int32,1}
# inputSize::Int32
# outputSize::Int32
# mins::Array{Float64,1}
# maxes::Array{Float64,1}
# means::Array{Float64,1}
# ranges::Array{Float64,1}