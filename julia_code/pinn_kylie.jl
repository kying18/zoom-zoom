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
using Plots
using DiffEqFlux
using Optim
using OrdinaryDiffEq
using Plots
using Statistics
using DiffEqSensitivity
using StaticArrays
using Flux

include("./bicycle_model.jl")
include("./generate_data.jl")

# 19 inputs for each input item in our x vector (7), u vector (2 - steer, D), p vector (10)
# 7 outputs for each output item in the newly predicted y vector (ignoring theta)

# rn, these are just dense networks. not sure that's the way to go.. tbd...
# 3 hidden layers.. should be able to approximate any fxn.. maybe architecture will come later
# we can ask chris i guess
nn = Chain(
    Dense(19, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 7),
)


# solve the bicycle model to get some data
# for this part let's assume no u, x0, y0= 0
p_ex=[350.0,3.0,1.5,1.5,500.0,21000.0,3430.0,1.2,-0.4,9.8] # only change the 5th, 6th, and 9th (mom of inertia, cornering, cla)
u0=[0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0].+ 2 .*(rand(Float64, (9)).-0.5) # commands appended to state

# prob = ODEProblem(bicycle_model, )
tspan = (0.0, 50.0)
prob=ODEProblem(diffeq_bicycle_model,u0,tspan,p_ex) #doesn't set dt and uses adaptive time steping
sol = solve(prob,Tsit5(),saveat=0.001)
plot(sol, vars=(1,2))

########### loss functions ########
# the reason why i have 2 is because the x, y version is what gets
# called from flux.train! but in order to see the loss from the callback
# we need a version that doesnt need the params
###################################
function loss()
  loss = 0
  for i in 2:length(sol.u)
    nn_pred = nn(vcat(sol.u[i-1], p_ex))
    actual = sol.u[i][1:7]
    loss += sum((nn_pred - actual).^2)
  end
  return loss
end
loss(x, y) = sum((nn(x) - y).^2)

########### setting up some stuff ########
lr = 0.01
opt = Flux.ADAM(lr)
# data = Iterators.repeated((), 5000)
iter = 0

########### getting xs and ys ########
xs = [(vcat(sol.u[i], p_ex)) for i=1:length(sol.u)-1]
ys = [sol.u[i][1:7] for i=2:length(sol.u)]
data = zip(xs, ys)

########### training the neural net ########
cb = function() #callback function to observe training
    global iter += 1
    if iter % 500 == 0
        display(loss())
    end
end
display(loss())
Flux.train!(loss, Flux.params(nn), data, opt; cb=cb)

########### plotting the results of the nn ########
plot(sol, vars=(1,2))
x = rand(length(xs)-1)
y = rand(length(xs)-1)
for i in 1:length(x)
    res = nn(xs[i])
    x[i] = res[1]
    y[i] = res[2]
end
plot!(x, y)








## MARK's SECTION
################ try generate_data.jl and single-step version for training #####################
function generate_training_data(t_num,u_inp_len=9,u_out_len=7,p_len=10)
  ### t_num is the number of training data pairs to use
  ### define all 3 training blocks
  training_block = [zeros(u_inp_len + p_len,t_num), zeros(u_out_len,t_num), zeros(u_out_len,t_num)]
  out_block_reg = zeros(u_out_length,training_set_num)
  out_block_rand = zeros(u_out_length,training_set_num)
  ###
  for i in 1:t_num
    # randomize u_0
    local u0_ex_gen = [0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0].+ 2 .*(rand(Float64, (9)).-0.5)

    #p[5] (cornering_stiff), p[9] (cla), p[5] (Iz) can change!
    local p_ex_gen=[350.0,3.0,1.5,1.5,550.0*(1.1 - 0.2*rand(Float64,1)[1]),10000.0*(1.1 - 0.2*rand(Float64,1)[1]),3430.0,1.2,-0.5*(1.2 - 0.4*rand(Float64,1)[1]),9.8]

    # get ideal training norm loss
    out_real = bicycle_model(u0_ex_gen[1:7], p_ex_gen, u0_ex_gen[8:9])

    # get ideal training norm loss
    out_rand = out_real.*(ones(length(out_real),1).*1.02 - 0.04*rand(Float64,length(out_real)))

    # push into T block
    #
    # training_block is: [input (u_0 + p array), ideal output for reg, slightly rand. output]
    training_block[1][:,i] = vcat(u0_ex_gen,p_ex_gen)
    training_block[2][:,i] = out_real
    training_block[3][:,i] = out_rand
  end
  return training_block
end





dp_num = 4000
t_block = generate_training_data(dp_num)

# best way to squish training block into something readable by Flux
inp_train = [t_block[1][:,i] for i in 1:size(t_block[1],2)]
outp_train = [t_block[3][:,i] for i in 1:size(t_block[3],2)]

train_dat = zip(inp_train,outp_train)

###### loss function regularized with bicycle model
### syntax info found here: https://fluxml.ai/Flux.jl/stable/training/training/

lam = 0.5
#loss_with_BM_reg(x, y) = Flux.Losses.mse(nn(x), y) + lam*Flux.Losses.mse(nn(x), bicycle_model(x[1:7], x[10:end], x[8:9]))
#loss_with_BM_reg(x, y) = sum(abs2,nn(x) - y)
loss_with_BM_reg(x) = sum(abs2,nn(x) - bicycle_model(x[1:7], x[10:end], x[8:9]))

lr = 0.01
opt = Flux.ADAM(lr)
# data = Iterators.repeated((), 5000)
iter = 0

nn = Chain(
    Dense(19, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 7),
)


test_x = xs[rand(1:length(xs))]
test_y = nn(test_x)# ... create single batch of test data ...
evalcb() = @show(loss_with_BM_reg(test_x, test_y))
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    test_x = xs[rand(1:length(xs))]
    test_y = nn(test_x)# ... create single batch of test data ...
    display(loss_with_BM_reg(test_x, test_y))
  end
end
display(loss_with_BM_reg(test_x))

Flux.train!(loss_with_BM_reg, Flux.params(nn), train_dat, opt;
            cb=cb)

################################################################################




#dat_block = gen_data(training_set_num, p_ex_gen)
#plot(dat_block[1,:])
# input
#input_block = [dat_block[i,:] for i=1:size(dat_block)[1]]
# output
#output_block = [dat_block[i,:] for i=1:size(dat_block)[1]]

@TODO # figure out gen_data! did not get enough info from Emily
#training_data = zip(input_block,output_block)
###################### try my own gen for testing ####################################################

#u_next = dormandprince(bicycle_model,u0,p_ex_gen,0.01)
#dormandprince(f, u, p, command, dt)


####################################################################################################################
## END OF MARK's SECTION



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
