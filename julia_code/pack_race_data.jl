cd(@__DIR__)
using Pkg
Pkg.activate(".")


using MAT, NumericalIntegration,ForwardDiff,FDM,FiniteDifferences


function unpack_MATs(mat_overpath="../roborace_data/SpeedGoat.mat")
    header = matopen(mat_overpath)

    # I was too lazy to
    # what do i do cornering stiffness measurements ...
    x_block = hcat(read(header, "OXT_E")[7356:13052,1],read(header, "OXT_E")[7356:13052,2]) # ``varname`` into scope
    y_block = hcat(read(header, "OXT_N")[7356:13052,1],read(header, "OXT_N")[7356:13052,2])
    vx_block = hcat(read(header, "OXT_vxCG")[7356:13052,1],read(header, "OXT_vxCG")[7356:13052,2])
    vy_block = hcat(read(header, "OXT_vyCG")[7356:13052,1],read(header, "OXT_vyCG")[7356:13052,2])
    psi_block = hcat(read(header, "MAP_psiDes")[1:10:end,1][7356:13052],read(header, "MAP_psiDes")[1:10:end,2][7356:13052])
    psi_block_full = read(header, "MAP_psiDes")
    r_block = hcat(psi_block[:,1],moving_avg(centraldiff(psi_block_full),50)[1:10:end][7356:13052])


    D_block = hcat(read(header, "CMD_gLongRequest")[1:10:end,1][7356:13052],read(header, "CMD_gLongRequest")[1:10:end,2][7356:13052])
    #steer_block = hcat(psi_block[:,1],atan.(2*tan.(atan.(vy_block[:,2]./vx_block[:,2])-psi_block[:,2])))
    steer_block = hcat(read(header, "DCU_aWheelFMean")[1:10:end,1][7356:13052],read(header, "DCU_aWheelFMean")[1:10:end,2][7356:13052])
    Delta_block = hcat(psi_block[:,1],moving_avg(centraldiff(steer_block),50))

    #(1)x, (2)y, (3)psi, (4)vx, (5)vy, (6)r, (7)steer, (8)D, (9) Delta
    #                                                ^^comm(1), comm(2)
    u_block = [x_block, y_block, psi_block, vx_block, vy_block, r_block, steer_block, D_block, Delta_block]

    if_on = read(header, "MAP_UxDes")

    return u_block
end




function unpack_MATs_no_time(mat_overpath="../roborace_data/SpeedGoat.mat")
    header = matopen(mat_overpath)

    # I was too lazy to
    # what do i do cornering stiffness measurements ...
    x_block = hcat(read(header, "OXT_E")[7356:13052,1],read(header, "OXT_E")[7356:13052,2]) # ``varname`` into scope
    y_block = hcat(read(header, "OXT_N")[7356:13052,1],read(header, "OXT_N")[7356:13052,2])
    vx_block = hcat(read(header, "OXT_vxCG")[7356:13052,1],read(header, "OXT_vxCG")[7356:13052,2])
    vy_block = hcat(read(header, "OXT_vyCG")[7356:13052,1],read(header, "OXT_vyCG")[7356:13052,2])
    psi_block = hcat(read(header, "MAP_psiDes")[1:10:end,1][7356:13052],read(header, "MAP_psiDes")[1:10:end,2][7356:13052])
    psi_block_full = read(header, "MAP_psiDes")
    r_block = hcat(psi_block[:,1],moving_avg(centraldiff(psi_block_full),50)[1:10:end][7356:13052])


    D_block = hcat(read(header, "CMD_gLongRequest")[1:10:end,1][7356:13052],read(header, "CMD_gLongRequest")[1:10:end,2][7356:13052])
    #steer_block = hcat(psi_block[:,1],atan.(2*tan.(atan.(vy_block[:,2]./vx_block[:,2])-psi_block[:,2])))
    steer_block = hcat(read(header, "DCU_aWheelFMean")[1:10:end,1][7356:13052],moving_avg(read(header, "DCU_aWheelFMean")[1:10:end,2][7356:13052],20))
    Delta_block = hcat(psi_block[:,1],moving_avg(centraldiff(steer_block),50))


    x_block = x_block[:,2]
    y_block = y_block[:,2]
    psi_block = psi_block[:,2]

    vx_block = vx_block[:,2]
    vy_block = vy_block[:,2]
    r_block = r_block[:,2]
    steer_block = steer_block[:,2]
    D_block = D_block[:,2]
    Delta_block = Delta_block[:,2]

    # remove reset spikes
    for i in 1:length(r_block)
        if abs(r_block[i]) > 5
            r_block[i] = 0
        end
    end

    vy_block = moving_avg(vy_block,20)
    r_block = moving_avg(vy_block,5)

    #(1)x, (2)y, (3)psi, (4)vx, (5)vy, (6)r, (7)steer, (8)D, (9) Delta
    #                                                ^^comm(1), comm(2)
    u_block = [x_block, y_block, psi_block, vx_block, vy_block, r_block, steer_block, D_block, Delta_block]


    return u_block
end





# how to call it
gg = unpack_MATs()
##




# take rough derivative
function centraldiff(v::AbstractArray)
    diff = Array{Float64}(undef,length(v[:,1]))
    for i in 1:(length(diff)-1)
        diff[i] = (v[i+1,2] - v[i,2])/(v[i+1,1] - v[i,1])
    end
    return diff
end




function moving_avg(X::Vector,numofele::Int)
    BackDelta = div(numofele,2)
    ForwardDelta = isodd(numofele) ? div(numofele,2) : div(numofele,2) - 1
    len = length(X)
    Y = similar(X)
    for n = 1:len
        lo = max(1,n - BackDelta)
        hi = min(len,n + ForwardDelta)
        Y[n] = mean(X[lo:hi])
    end
    return Y
end










header = matopen(mat_overpath)

# I was too lazy to
# what do i do cornering stiffness measurements ...
x_block = hcat(read(header, "OXT_E")[7356:13052,1],read(header, "OXT_E")[7356:13052,2]) # ``varname`` into scope
y_block = hcat(read(header, "OXT_N")[7356:13052,1],read(header, "OXT_N")[7356:13052,2])
vx_block = hcat(read(header, "OXT_vxCG")[7356:13052,1],read(header, "OXT_vxCG")[7356:13052,2])
vy_block = hcat(read(header, "OXT_vyCG")[7356:13052,1],read(header, "OXT_vyCG")[7356:13052,2])
psi_block = hcat(read(header, "MAP_psiDes")[1:10:end,1][7356:13052],read(header, "MAP_psiDes")[1:10:end,2][7356:13052])
psi_block_full = read(header, "MAP_psiDes")
r_block = hcat(psi_block[:,1],moving_avg(centraldiff(psi_block_full),50)[1:10:end][7356:13052])


D_block = hcat(read(header, "CMD_gLongRequest")[1:10:end,1][7356:13052],read(header, "CMD_gLongRequest")[1:10:end,2][7356:13052])
#steer_block = hcat(psi_block[:,1],atan.(2*tan.(atan.(vy_block[:,2]./vx_block[:,2])-psi_block[:,2])))
steer_block = hcat(read(header, "DCU_aWheelFMean")[1:10:end,1][7356:13052],moving_avg(read(header, "DCU_aWheelFMean")[1:10:end,2][7356:13052],20))
Delta_block = hcat(psi_block[:,1],moving_avg(centraldiff(steer_block),50))


x_block = x_block[:,2]
y_block = y_block[:,2]
psi_block = psi_block[:,2]

vx_block = vx_block[:,2]
vy_block = vy_block[:,2]
r_block = r_block[:,2]
steer_block = steer_block[:,2]
D_block = D_block[:,2]
Delta_block = Delta_block[:,2]

# remove reset spikes
for i in 1:length(r_block)
    if abs(r_block[i]) > 5
        r_block[i] = 0
    end
end

vy_block = moving_avg(vy_block,20)
r_block = moving_avg(vy_block,5)


plot(steer_block)
plot!(Delta_block)
