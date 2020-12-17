cd(@__DIR__)
using Pkg
Pkg.activate(".")


using MAT, NumericalIntegration,ForwardDiff,FDM,FiniteDifferences


function unpack_MATs(mat_overpath="../roborace_data/SpeedGoat.mat")
    header = matopen(mat_overpath)
    # all cornering stiffness measurements i'm assuming
    x_block = read(header, "OXT_E") # ``varname`` into scope
    y_block = read(header, "OXT_N")
    vx_block = read(header, "OXT_vxCG")
    vy_block = read(header, "OXT_vyCG")
    psi_block = hcat(read(header, "MAP_psiDes")[1:10:end,1],read(header, "MAP_psiDes")[1:10:end,2])
    r_block = hcat(psi_block[:,1],centraldiff(psi_block))

    steer_accel_block = read(header, "CMD_aWheelFSteer")
    rate_block = Array(cumul_integrate(steer_accel_block[:,1], steer_accel_block[:,2]))
    steer_block = hcat(steer_accel_block[:,1][1:10:end],cumul_integrate(stear_accel_block[:,1],rate_block)[1:10:end])

    D_block = hcat(read(header, "CMD_gLongRequest")[1:10:end,1],read(header, "CMD_gLongRequest")[1:10:end,2])
    foo1 = read(header, "CMD_aWheelFSteer")[:,1]
    foo2 = read(header, "CMD_aWheelFSteer")[:,2]
    Delta_block = hcat(foo1[1:10:end],cumul_integrate(foo1,foo2)[1:10:end])


    #(1)x, (2)y, (3)psi, (4)vx, (5)vy, (6)r, (7)steer, (8)D, (9) Delta
    u_block = [x_block, y_block, psi_block, vx_block, vy_block, r_block, steer_block, D_block, Delta_block]

    return u_block
end

gg = unpack_MATs()




function centraldiff(v::AbstractArray)
    diff = Array{Float64}(undef,length(v[:,1]))
    diff[1] = 0
    for i in 2:(length(diff)-1)
        diff[i] = (v[i+1,2] - v[i,2])/(v[i+1,1] - v[i,1])
    end

    return diff
end
