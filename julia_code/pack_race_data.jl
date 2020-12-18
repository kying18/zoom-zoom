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
    r_block = hcat(psi_block[:,1],centraldiff(psi_block_full)[1:10:end][7356:13052])


    D_block = hcat(read(header, "CMD_gLongRequest")[1:10:end,1][7356:13052],read(header, "CMD_gLongRequest")[1:10:end,2][7356:13052])
    steer_block = hcat(psi_block[:,1],atan.(2*tan.(atan.(vy_block[:,2]./vx_block[:,2])-psi_block[:,2])))
    Delta_block = hcat(psi_block[:,1],centraldiff(steer_block))

    #(1)x, (2)y, (3)psi, (4)vx, (5)vy, (6)r, (7)steer, (8)D, (9) Delta
    #                                                ^^comm(1), comm(2)
    u_block = [x_block, y_block, psi_block, vx_block, vy_block, r_block, steer_block, D_block, Delta_block]

    if_on = read(header, "MAP_UxDes")

    return u_block
end

gg = unpack_MATs()



# take rough derivative
function centraldiff(v::AbstractArray)
    diff = Array{Float64}(undef,length(v[:,1]))
    for i in 1:(length(diff)-1)
        diff[i] = (v[i+1,2] - v[i,2])/(v[i+1,1] - v[i,1])
    end
    return diff
end


if_on = hcat(read(header, "MAP_UxDes")[1:10:end,1],read(header, "MAP_UxDes")[1:10:end,2])

if_bool = filter(x->x>0.0, if_on[:,2])
