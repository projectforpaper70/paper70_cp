using IntervalArithmetic,SetRounding
export weight2interval32, bias2interval32, input2interval32
export weight2interval16, bias2interval16, input2interval16
#权重和偏重区间化

function weight2interval32(scalar_weight::Array{T,2}) where {T<:Real}

    RATIONAL_ABS = BigFloat(2^-149)
    RATIONAL_REL_UP = 1 + BigFloat(2^-20)
    RATIONAL_REL_LO = 1 - BigFloat(2^-20)
    INTERVAL_REL = Interval(RATIONAL_REL_LO ,RATIONAL_REL_UP )
    INTERVAL_ABS = Interval(-RATIONAL_ABS , RATIONAL_ABS)
    ####对于64位和32位统一用32位表示
    # println("这里权重的数据类型是Float#设置输入约束
    setprecision(BigFloat,23)
    Itv_weight=similar(scalar_weight, Interval{Float64})
    for i in 1:size(scalar_weight,1)
        for j in 1:size(scalar_weight,2)

            # setrounding(BigFloat, RoundUp)
            # setrounding(Float32,RoundUp)
            # ub = Float32(BigFloat(scalar_weight[i,j]))
            # setrounding(Float32,RoundUp)
            # ub = Float32(scalar_weight[i,j])

            # setrounding(BigFloat, RoundDown)
            # setrounding(Float32,RoundDown)
            # lb = Float32(BigFloat(scalar_weight[i,j]))
            # setrounding(Float32,RoundDown)
            # lb = Float32(scalar_weight[i,j])


            #println(lb,fc1_weight[i,j],ub)
            Itv_weight[i,j]=scalar_weight[i,j]*INTERVAL_REL+INTERVAL_ABS
            @assert(scalar_weight[i,j]<=Itv_weight[i,j].hi && scalar_weight[i,j]>=Itv_weight[i,j].lo, "the lb of weight: $(Itv_weight[i,j].lo) great than ub: $(Itv_weight[i,j].hi)")
        end
    end
    return Itv_weight
end


function bias2interval32(scalar_bias::Array{T,1}) where {T<:Real}
    # println("typeof T is ",typeof(T))
    # println("the precison of bigfloat32 is ",precision(BigFloat))

    ####对于64位和32位统一用32位表示
    # println("这里权重的数据类型是Float16\n")

    setprecision(BigFloat,23)
    Itv_bias=similar(scalar_bias, Interval{Float64})
    for i in 1:size(scalar_bias,1)
        RATIONAL_ABS = BigFloat(2^-149)
        RATIONAL_REL_UP = 1 + BigFloat(2^-20)
        RATIONAL_REL_LO = 1 - BigFloat(2^-20)
        INTERVAL_REL = Interval(RATIONAL_REL_LO ,RATIONAL_REL_UP )
        INTERVAL_ABS = Interval(-RATIONAL_ABS , RATIONAL_ABS)
        # setrounding(BigFloat, RoundUp)
        # setrounding(Float32,RoundUp)
        # ub = Float32(BigFloat(scalar_bias[i]))
        # setrounding(Float32,RoundUp)
        # ub = Float32(scalar_bias[i])


        # setrounding(BigFloat, RoundDown)
        # setrounding(Float32,RoundDown)
        # lb = Float32(BigFloat(scalar_bias[i]))
        # setrounding(Float32,RoundDown)
        # lb = Float32(scalar_bias[i])


        Itv_bias[i] = scalar_bias[i]*INTERVAL_REL+INTERVAL_ABS
        @assert(scalar_bias[i]<=Itv_bias[i].hi && scalar_bias[i]>=Itv_bias[i].lo, "the lb of bias: $(Itv_bias[i].lo) the ub of bias: $(Itv_bias[i].hi) ,bias is $(bias[i])")
    end
    return Itv_bias

    ##16位编码结束######
end

#输入区间化
function input2interval32(scalar_input::Array{T,1}) where {T<:Real}
    # println("typeof T is ",typeof(T))
    # println("the precison of bigfloat32 is ",precision(BigFloat))

    ####对于64位和32位统一用32位表示
    # println("这里权重的数据类型是Float16\n")

    setprecision(BigFloat,23)
    Itv_bias=similar(scalar_input, Interval{Float32})
    for i in 1:size(scalar_input,1)
        setrounding(BigFloat, RoundUp)
        ub = Float32(BigFloat(scalar_input[i])) 
        setrounding(BigFloat, RoundDown)
        lb = Float32(BigFloat(scalar_input[i])) 
        Itv_bias[i]=interval(lb,ub)
    end
    return Itv_bias

    ##16位编码结束######
end




#权重和偏重区间化

function weight2interval16(scalar_weight::Array{T,2}) where {T<:Real}
    #println("typeof T is ",typeof(T))
    #println("the precison of bigfloat32 is ",precision(BigFloat))

    ####对于64位和32位统一用16位表示
    # println("这里权重的数据类型是Float#设置输入约束
    #setprecision(BigFloat,10)
    Itv_weight=similar(scalar_weight, Interval{Float64})
    for i in 1:size(scalar_weight,1)
        for j in 1:size(scalar_weight,2)

            RATIONAL_ABS = Float64(2^-13)
            RATIONAL_REL_UP = 1 + Float64(2^-10)
            RATIONAL_REL_LO = 1 - Float64(2^-10)
            INTERVAL_REL = Interval(RATIONAL_REL_LO ,RATIONAL_REL_UP )
            INTERVAL_ABS = Interval(-RATIONAL_ABS , RATIONAL_ABS)

            # setrounding(BigFloat, RoundUp)
            # setrounding(Float64,RoundUp)
            # ub = Float64(BigFloat(scalar_weight[i,j]))
            # # setrounding(BigFloat, RoundUp)
            # # ub = Float16(BigFloat(scalar_weight[i,j])) 

            
            # setrounding(BigFloat, RoundDown)
            # setrounding(Float64,RoundDown)
            # lb = Float64(BigFloat(scalar_weight[i,j]))
            # # setrounding(BigFloat, RoundDown)
            # # lb = Float16(BigFloat(scalar_weight[i,j])) 
            # #println(lb,fc1_weight[i,j],ub)
            Itv_weight[i,j]=scalar_weight[i,j]*INTERVAL_REL+INTERVAL_ABS
            #@assert(scalar_weight[i,j]<=Itv_weight[i,j].hi && scalar_weight[i,j]>=Itv_weight[i,j].lo, "the lb of weight: $(Itv_weight[i,j].lo) great than ub: $(Itv_weight[i,j].hi),weight is $(scalar_weight[i,j])")
            # println(showfull(Itv_fc1_weight[i,j]))
            # print("\n")
        end
    end
    return Itv_weight
end


function bias2interval16(scalar_bias::Array{T,1}) where {T<:Real}
    # println("typeof T is ",typeof(T))
    # println("the precison of bigfloat32 is ",precision(BigFloat))

    ####这里再尝试使用16位浮点编码，效率可能会提升
    # println("这里权重的数据类型是Float16\n")

    #setprecision(BigFloat,12)
    Itv_bias=similar(scalar_bias, Interval{Float64})
    for i in 1:size(scalar_bias,1)
        RATIONAL_ABS = Float64(2^-13)
        RATIONAL_REL_UP = 1 + Float64(2^-10)
        RATIONAL_REL_LO = 1 - Float64(2^-10)
        INTERVAL_REL = Interval(RATIONAL_REL_LO ,RATIONAL_REL_UP )
        INTERVAL_ABS = Interval(-RATIONAL_ABS , RATIONAL_ABS)
        # setrounding(BigFloat, RoundUp)
        # setrounding(Float64,RoundUp)
        # ub = Float64(BigFloat(scalar_bias[i]))
        # setrounding(BigFloat, RoundUp)
        # ub = Float16(BigFloat(scalar_bias[i])) 

        # setrounding(BigFloat, RoundDown)
        # setrounding(Float64,RoundDown)
        # lb = Float64(BigFloat(scalar_bias[i]))
        # setrounding(BigFloat, RoundDown)
        # lb = Float16(BigFloat(scalar_bias[i])) 
        # Itv_bias[i]=interval(lb,ub)
        Itv_bias[i] = scalar_bias[i]*INTERVAL_REL+INTERVAL_ABS
        #@assert(scalar_bias[i]<=Itv_bias[i].hi && scalar_bias[i]>=Itv_bias[i].lo, "the lb of bias: $(Itv_bias[i].lo) the ub of bias: $(Itv_bias[i].hi) ,bias is $(bias[i])")
    end
    return Itv_bias

    ##16位编码结束######
end

#输入区间化
function input2interval16(scalar_input::Array{T,1}) where {T<:Real}
    # println("typeof T is ",typeof(T))
    # println("the precison of bigfloat32 is ",precision(BigFloat))

    ####这里再尝试使用16位浮点编码，效率可能会提升
    # println("这里权重的数据类型是Float16\n")

    setprecision(BigFloat,10)
    Itv_bias=similar(scalar_input, Interval{Float16})
    for i in 1:size(scalar_input,1)
        setrounding(BigFloat, RoundUp)
        ub = Float16(BigFloat(scalar_input[i])) 
        setrounding(BigFloat, RoundDown)
        lb = Float16(BigFloat(scalar_input[i])) 
        Itv_bias[i]=interval(lb,ub)
    end
    return Itv_bias

    ##16位编码结束######
end

