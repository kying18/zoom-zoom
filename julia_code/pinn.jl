cd(@__DIR__)
using Pkg
Pkg.activate(".")
# Pkg.add("DiffEqFlux")
# Pkg.add("Flux")
# Pkg.add("Optim")
# Pkg.add("OrdinaryDiffEq")
# Pkg.add("Plots")
# Pkg.add("Statistics")
# Pkg.add("DiffEqSensitivity")
using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, Plots, Statistics, DiffEqSensitivity,StaticArrays
using MAT

include("./bicycle_model.jl")









########################## Get and Pack MAT data into training obj #######################

mutable struct bm_params
  u::Float64
  t::Float64
  y::Float64
  h::Float64
  z::Float64
  loss::Float64
  bm_params() = new()
end

mutable struct training_set
  time_series_block::Array{Float64,2}
  training_set() = new()
end

mat_overpath = "../roborace_data/SpeedGoat.mat"
header = matopen(mat_overpath)
alpha_cs_front = read(header, "TireSlip_alphaFront_rad") # ``varname`` into scope
function unpack_MAT(mat_overpath::String)
    header = matopen(mat_overpath)
    # all cornerist stiffness measurements i'm assuming
    alpha_cs_front = read(header, "TireSlip_alphaFront_rad") # ``varname`` into scope
    alpha_cs_back = read(header, "tireSlip_alphaRear_rad")
    kap_cs_front = read(header, "tireSlip_kappaFront_unitless")
    kap_cs_back = read(header, "tireSlip_kappaRear_unitless")
    wtf_is_this_front = read(header, "tireSlip_kappaRear_unitless") # idk what these vals are
    wtf_is_this_back = read(header, "tireSlip_kappaRearInd_unitless") # idk what these vals are
    phi_h = read(header, "NAV_aHeading")
    yaw_r = read(header, "NAV_aYaw")

    @Todo #figure out what to do with this block structuring
    p_block = hcat(alpha_cs_front,alpha_cs_back,kap_cs_front,kap_cs_back,phi_h,yaw_r)
end


##################################################################################

############################### Generate regularizer #############################


# I think it's just this simple. I guess we gotta try
function loss_and_regularizer(NN_out,inp, p, command=nothing)
    reg = bicycle_model!(du, inp, p, command)
    return sum(abs2,NN_out - reg)
end




#################


# 21 inputs for each input item in our x vector, u vector, p vector
# 8 outputs for each output item in the newly predicted y vector

# rn, these are just dense networks. not sure that's the way to go.. tbd...
# 3 hidden layers.. should be able to approximate any fxn.. maybe architecture will come later
# we can ask chris i guess
nn = Chain(
    Dense(21, 128, tanh),
    Dense(128, 128, tanh),
    Dense(128, 128, tanh),
    Dense(128, 8),
)


#define states in vect with dims as:
# u =
    #1| X    - x pos
    #2| Y    - y pos
    #3| phi  - steering angle
    #4| v_x  - x vel (w.r.t. body)
    #5| v_y  - y vel (w.r.t. body)
    #6| r    - yaw rate
    #7| del  - steering angle
    #8| T    - driver command

############ L2 Loss based on state params #############
function loss_L2(u)
l_r = 2 # set rear axel length
l_f = 2 # set front axel length

I_z = 2 # set moment of inertia
contr_gain = 1 # TODO define controller gain

r_targ = u[7]*(u[4]/(l_f + l_r))
tau_tv = (r_targ - u[6])*contr_gain

# define bicycle model ODE
u_der = @MVector[0.0 for i = 1:8]

u_der[1] = (u[4]*cos(u[3]))-(u[5]*sin(u[3]))
u_der[2] = (u[4]*cos(u[3]))-(u[5]*sin(u[3]))
u_der[3] = r
u_der[4] = (m^(-1))*(F_x - F_ry*sin(u[7]) + m*u[5]*u[6])
u_der[5] = (m^(-1))*(F_ry + F_ry*cos(u[7]) - m*u[4]*u[6])
u_der[6] = (I_z^(-1))*(F_Fy*l_f*cos(u[7]) - F_ry*l_r - m*u[4]*u[6])
u_der[7] = del_del # TODO ignore external commands
u_der[8] = del_T # TODO ignore external commands

# i think we can probably use the bicycle model defined in bicycle_model.jl?
# then you can do solve_bicycle_model (see below) - actual x here to calculate loss fxn

solve_RK(u_der, init_params,step_iter=1) # TODO implement RK solver for this

end

function solve_bicycle_model(x, u, p, t)
  u = [x ; u]  # TODO: figure out dimensionality here (whehter this should be vcat or hcat)

  # this u will be a vector of x (ie x, y, psi, etc.) and u (user input) (x)
  # x, y, psi, vx, vy, r, theta, steer = x
  # v_theta, D, delta = u  # velocity along path, accel command, commanded steer rate
  # u[1:8] = x, y, psi, vx, vy, r, theta, steer
  # u[9:11] = v_theta, D, delta

  tspan = (0.0, 1.0) # TODO.. make this one timestep?
  prob = ODEProblem(bicycle_model, u, tspan, p)  # this may not be properly imported, not sure
  solution = solve(prob, DP5(), saveat=0.25)

  # TODO: return last element of solution
  # not sure about syntax
  # maybe solution[5] ??

end


opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(loss_L2())
  end
end
display(loss_L2())

Flux.train!(loss_L2, Flux.params(nn), data, opt; cb=cb)
