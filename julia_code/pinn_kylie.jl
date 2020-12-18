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
include("./pack_race_data.jl")

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
using LinearAlgebra
function generate_training_data(t_num,u_inp_len=9,u_out_len=7,p_len=10,dt=0.04)
  ### t_num is the number of training data pairs to use
  ### define all 3 training blocks
  training_block = [zeros(u_inp_len + p_len,t_num), zeros(u_out_len,t_num), zeros(u_out_len,t_num)]
  out_block_reg = zeros(u_out_length,training_set_num)
  out_block_rand = zeros(u_out_length,training_set_num)
  ###
  for i in 1:t_num
    # randomize u_0
    local u0_ex_gen = vcat([0.0,0.0,0.0,5.0,0.0,0.0,0.0].+ 2 .*(rand(Float64, (7)).-0.5),[0.0,0.0])
    #local u0_ex_gen = foo/norm(foo)

    #p[5] (cornering_stiff), p[9] (cla), p[5] (Iz) can change!
    local p_ex_gen = [350.0,3.0,1.5,1.5,550.0*(1.1 - 0.2*rand(Float64,1)[1]),10000.0*(1.02 - 0.04*rand(Float64,1)[1]),3430.0,1.2,-0.5*(1.02 - 0.04*rand(Float64,1)[1]),9.8]
    #local p_ex_gen = foo2/norm(foo2)
    # get ideal training norm loss
    #foo2 = bicycle_model(u0_ex_gen[1:7], p_ex_gen, u0_ex_gen[8:9])

    # push into a DP solver to get the next u output for some fixed timestep
    out_real = dormandprince(bicycle_model, u0_ex_gen[1:7], p_ex_gen, u0_ex_gen[8:9], dt)
    #out_real = foo2/norm(foo2) # normalize the real outputs

    # get ideal training norm loss
    out_rand = out_real.*(ones(length(out_real),1).*1.02 - 0.04*rand(Float64,length(out_real)))
    #out_rand = foo3/norm(foo3) # normalize the rand outputs
    # push into T block
    #
    # training_block is: [input (u_0 + p array), ideal output for reg, slightly rand. output]
    training_block[1][:,i] = vcat(u0_ex_gen,p_ex_gen)/norm(vcat(u0_ex_gen,p_ex_gen))
    training_block[2][:,i] = out_real
    training_block[3][:,i] = out_rand
  end
  return training_block
end






########generate race data training block################
u_raw = unpack_MATs()

function generate_race_data(t_num,u_inp_len=9,u_out_len=7,p_len=10,dt=0.04)

  u_raw = unpack_MATs()
  foox = u_raw[1]
  fooy = u_raw[2]

  x_step = copy(u_raw)

  checkerx = copy(foox)
  checkery = copy(fooy)

  u_train_inp = Vector{Array{Float64,1}}(undef,length(foox[:,2])-1)
  u_train_out = Vector{Array{Float64,1}}(undef,length(foox[:,2])-1)


  for i in 1:(length(foox[:,2])-1)
    u_train_inp[i] = [u_raw[j][i,2] for j in 1:9]
    u_train_out[i] = [u_raw[j][i+1,2] for j in 1:9]
    u_train_out[i][1] = u_raw[1][i+1,2] -  u_raw[1][i,2]
    u_train_out[i][2] = u_raw[2][i+1,2] -  u_raw[2][i,2]

    u_train_inp[i][1] = 0
    u_train_inp[i][2] = 0

    deleteat!(u_train_inp[i], 1:2)


  end
  #(1)x, (2)y, (3)psi, (4)vx, (5)vy, (6)r, (7)steer, (8)D, (9) Delta
  #                                                ^^comm(1), comm(2)
  return zip(u_train_inp,u_train_out), u_train_inp, u_train_out
end






function generate_race_data_no_time(t_num,u_inp_len=9,u_out_len=7,p_len=10,dt=0.04)

  u_raw = unpack_MATs_no_time()
  foox = u_raw[1]
  fooy = u_raw[2]

  x_step = copy(u_raw)

  checkerx = copy(foox)
  checkery = copy(fooy)

  start_t_batch = 1200
  u_train_inp = Vector{Array{Float64,1}}(undef,length(foox)-1-start_t_batch)
  u_train_out = Vector{Array{Float64,1}}(undef,length(foox)-1-start_t_batch)

  for n in start_t_batch+1:(length(foox)-1)
    i = n - start_t_batch
    u_train_inp[i] = [u_raw[j][i] for j in 3:9]
    u_train_out[i] = [u_raw[j][i+1] for j in 1:2]
    u_train_out[i][1] = u_raw[1][i+1] -  u_raw[1][i]
    u_train_out[i][2] = u_raw[2][i+1] -  u_raw[2][i]
    #u_train_out[i][3] = u_raw[3][i+1] -  u_raw[3][i]

  end


  #u_train_input = u_train_inp./norm(u_train_inp)


  #(1)x, (2)y, (3)psi, (4)vx, (5)vy, (6)r, (7)steer, (8)D, (9) Delta
  #                                                ^^comm(1), comm(2)
  return zip(u_train_inp,u_train_out), u_train_inp, u_train_out
end



####################################################









######################## RACE DATA


train_dat_race, ut_inp, ut_out =  generate_race_data_no_time(6000)



#############################################################################################
########################### Try Emily's Data Generation #####################################

# @TODO
#####################################################################################################


###### loss function regularized with bicycle model
### syntax info found here: https://fluxml.ai/Flux.jl/stable/training/training/



nn = Chain(
    Dense(7, 14, tanh),
    Dropout(0.00001),
    Dense(14, 14, tanh),
    Dropout(0.00001),
    Dense(14, 2),
)





##################################

lam = 2
dt = 0.01
#loss_with_BM_reg(x, y) = Flux.Losses.mse(nn(x), y) + lam*Flux.Losses.mse(nn(x), bicycle_model(x[1:7], x[10:end], x[8:9]))
#loss_with_BM_reg(x, y) = sum(abs2,nn(x) - y)
#loss_with_BM_only(x,y) = Flux.Losses.mse(nn(x), y) + lam*Flux.Losses.mse(nn(x), training_block[2][:,findall(seek->seek==x,inp_train)])

#loss_with_BM(x,y) = Flux.Losses.mse(nn(x)[3:end], y[3:end]) + lam*Flux.Losses.mse(nn(x)[1:2], y[1:2])
loss_with_BMM(x,y) = Flux.Losses.mse(nn(x), y)


lr = 0.0001
opt = Flux.ADAM(lr)
# data = Iterators.repeated((), 5000)


# IMPORTANT: set a large enough batch size for the test data
# Basically I do it by randomizing a batch pick from the training
# set and just add up the total loss from that batch
batch_size = 1000
iter = 0
cb = function ()
  global iter += 1
  if iter % 200 == 0
    loss_counter = 0
    for iter in 1:batch_size
      foo = rand(1:length(ut_inp))
      test_x = ut_inp[foo]
      test_y = ut_out[foo]
      loss_counter += loss_with_BMM(test_x,test_y)
    end
    display(loss_counter)

  end
end

testmode!(nn, false)
using Flux: @epochs
@epochs 500 Flux.train!(loss_with_BMM, Flux.params(nn), train_dat_race, opt;
            cb=cb)

################################################################################
##################### let's try it out #########################################


#tester_inp = copy(ut_inp)
#tester_out = copy(ut_out)
#x_traj = Vector{Float64}(undef,length(tester_inp))
#y_traj = Vector{Float64}(undef,length(tester_inp))


testmode!(nn, true)

foo = rand(1:length(outp_train))
test_x = ut_inp[foo]
test_y = ut_out[foo]



pp = generate_race_data_no_time(4000)


####################################################
x_coord = pp[3]
nn_inp = pp[2]

gq = zeros(length(x_coord))
nn_q = zeros(length(x_coord))
for i in 2:1:length(x_coord)
  gq[i] = gq[i-1] + x_coord[i][1]
  nn_q[i] = nn_q[i-1] + nn(nn_inp[i])[1]
end

y_coord = pp[3]
nn_g = pp[2]

gy = zeros(length(x_coord))
nn_y = zeros(length(x_coord))
for i in 2:1:length(x_coord)
  gy[i] = gy[i-1] + y_coord[i][2]
  nn_y[i] = nn_y[i-1] + nn(nn_g[i])[2]
end

plot(gq)
plot!(nn_q)



plot(gy)
plot!(nn_y.-200)

plot(gq[1:1000],gy[1:1000])
plot!(nn_q[1:1000],nn_y[1:1000]-200)





check = nn(test_x)

for i in 1:length(x_traj)

end

















#from orig dataset:

foo = rand(1:length(outp_train))
test_x = inp_train[foo]
test_y = outp_train[foo]
loss_counter = loss_with_BM_only(test_x,test_y)



nn(test_x)



foo = vcat([0.0,0.0,0.0,5.0,0.0,0.0,0.0].+ 2 .*(rand(Float64, (7)).-0.5),[0.0,0.0])
u0_ex_gen = foo/norm(foo)

#p[5] (cornering_stiff), p[9] (cla), p[5] (Iz) can change!
foo2 = [350.0,3.0,1.5,1.5,550.0*(1.1 - 0.2*rand(Float64,1)[1]),10000.0*(1.02 - 0.04*rand(Float64,1)[1]),3430.0,1.2,-0.5*(1.02 - 0.04*rand(Float64,1)[1]),9.8]
p_ex_gen = foo2/norm(foo2)

inp  = vcat(u0_ex_gen,p_ex_gen)
# get ideal training norm loss
#foo2 = bicycle_model(u0_ex_gen[1:7], p_ex_gen, u0_ex_gen[8:9])

# push into a DP solver to get the next u output for some fixed timestep
out_real = dormandprince(bicycle_model, foo[1:7], foo2, foo[8:9], 0.01)



nn(inp)

loss_with_BM_only(vcat(p_test,u0_test),out_test)




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
