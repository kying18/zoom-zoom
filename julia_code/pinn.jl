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

solve_RK(u_der, init_params,step_iter=1) # TODO implement RK solver for this

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
