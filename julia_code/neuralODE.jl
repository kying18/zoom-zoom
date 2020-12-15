cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux, DiffEqBase, OrdinaryDiffEq, Plots, DataStructures, DiffEqFlux, Zygote

U0=Float32[1.0,1.0,1.0,5.0,1.0,1.0,1.0,1.0,1.0] #9
PT=Float32[350.0,3.0,1.5,1.5,550.0,10000.0,3430.0,1.2,-0.5,9.8] #10
P=PT./[100,1,1,1,100,10000,1000,1,1,1] # normalization
Float32[3.5,3.0,1.5,1.5,5.5,1.0,3.43,1.2,-0.5,9.8] #10

include("./bicycle_model.jl")

datasize = 30
tspan = (0.0f0,1.5f0)

t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(diffeq_bicycle_model,U0,tspan,PT)
ode_data = Array(solve(prob,Tsit5(),saveat=t))


dudt = Chain(x->vcat(x,P),
             Dense(17,50,tanh),
             Dense(50,7))

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-4,abstol=1e-4) # decrease tolerances

ps = Flux.params(n_ode)

pred = n_ode(U0[1:7]) # Get the prediction using the correct initial condition
scatter(t,ode_data[1:3,:]',label="data")
scatter!(t,pred[1:3,:]',label="prediction")

function predict_n_ode()
  n_ode(U0[1:7])
end
loss_n_ode() = sum(abs2,Array(ode_data)[1:7,:].- predict_n_ode())



data = Iterators.repeated((), 1000)
opt = ADAM(0.002)
iter = 0
cb = function () #callback function to observe training
  if iter % 10 == 0
    display(loss_n_ode())
    # plot current prediction against data
    cur_pred = predict_n_ode()
    pl = scatter(t,ode_data[1:3,:]',label="data",color=:blue)
    scatter!(t,cur_pred[1:3,:]',label="prediction",color=:orange)
    display(plot(pl))
  end
  global iter += 1
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

pl = scatter(t,ode_data[1:3,:]',label=["data x" "data y" "data psi"])
cur_pred = predict_n_ode()
scatter!(t,cur_pred[1:3,:]',label=["pred x" "pred y" "pred psi"])
display(plot(pl))
png("neuralODE_train")


## let's now try to get this to train faster, on different data
using CUDA
if has_cuda()
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

#=
Examples for gen_data
=#
# Set minimum and maximum values for initial state, parameters, and commands
min_psi = 0.0   # Don't change
max_psi = 2.0*pi # Don't change
min_v = 0.0
max_v = 30.0            # 30.0 m/s = 108 km/hr
minmax_r = pi/128.0     # Min yaw rate (derivative of psi), pi/128 is about 3 deg
minmax_steer = 0.26     # Don't change, from Driverless
min_D = -20.0           # Driverless = [-20,15]
max_D = 15.0
minmax_delta = 30.0*pi/360.0 # Don't change. Max change in delta is 15 degrees,
min_m = 200.0   # Driverless = 350.0
max_m = 1000.0
min_l = 2.5     # Driverless = 3.0
max_l = 3.5
min_lflr = 1.0  # Driverless = 1.5
max_lflr = 2.0
min_Iz = 550.0  # Driverless = 550.0
max_Iz = 600.0
min_cornering_stiff = 20000.0   # Driverless = [20,000,50,000]
max_cornering_stiff = 50000.0
min_cla = -0.7	# Driverless = 0.5
max_cla = -0.3

# [psi, vx, vy, r, steer, D, delta, m, l, lf/lr, Iz, cornering_stiff, cla]
lb = [min_psi, min_v, min_v, -minmax_r, -minmax_steer, min_D, -minmax_delta, min_m, min_l, min_lflr, min_Iz, min_cornering_stiff, min_cla]
ub = [max_psi, max_v, max_v, minmax_r, minmax_steer, max_D, minmax_delta, max_m, max_l, max_lflr, max_Iz, max_cornering_stiff, max_cla]


function get_data_no_noise(num_data, lb, ub)
  data = gen_data(num_data, lb, ub)
  x = hcat(zeros(num_data, 2), data[:,1:7])

  # [psi0, vx0, vy0, r0, steer0, D, delta, cornering_stiff, x, y, psi, vx, vy, r, steer]
  # 1:7 inputs
  # 9:15 outputs

  return x, y
end

num_time_points = 10
tspan = (0.0f0,1.0f0)

datasize = 10

num_train = floor(Int, 0.8*datasize)
num_valid = floor(Int, 0.1*datasize)
num_test = floor(Int, 0.1*datasize)

dudt = Chain(x->vcat(x,P),
             Dense(17,50,tanh),
             Dense(50,7))

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-4,abstol=1e-4) # decrease tolerances

ps = Flux.params(n_ode)

xtrain, ytrain = get_data_no_noise(num_train, lb, ub)
t = range(tspan[1],tspan[2],length=num_time_points)
ode_data = Array{Float32, 2}(undef, 9, 0)
for i=1:num_train
  u0 = xtrain[i, :]
  prob = ODEProblem(diffeq_bicycle_model,U0,tspan,P)
  ode_data = hcat(ode_data, Array(solve(prob,Tsit5(),saveat=t)))
end

pl = scatter() # clears the plot
reds=range(colorant"lightsalmon", stop=colorant"red4", length=datasize)
blues=range(colorant"skyblue", stop=colorant"navy", length=datasize)
# pred = Array(n_ode(U0[1:7])) # Get the prediction using the correct initial condition
for i=1:num_train-1
  scatter!(t,ode_data[1:3,i*num_time_points:(i+1)*num_time_points-1]',label="",color=blues[i])
  # scatter!(t,ode_data[1:3,i*num_time_points:(i+1)*num_time_points-1]')
end
function predict_n_ode()
  preds = Array{Float32, 2}(undef, 7, 0)
  for i=1:num_train
    pred = Array(n_ode(xtrain[i, 1:7]))
    # scatter!(t,pred[1:3,:]',label="",color=reds[i])
    preds = hcat(preds, pred)
  end
  return preds
end
loss_n_ode() = 1/(num_time_points * datasize) * sum(abs2,Array(ode_data)[1:7,:].- predict_n_ode())
predict_n_ode()
scatter!()
loss_n_ode()

data = Iterators.repeated((), 1000)
opt = ADAM(0.003)
iter = 0
cb = function () #callback function to observe training
  if iter % 10 == 0
    display(loss_n_ode())
    # plot current prediction against data
    pl = scatter()
    for i=1:num_train-1
      scatter!(t,ode_data[1:3,i*num_time_points:(i+1)*num_time_points-1]',label="",color=blues[i])
      pred = Array(n_ode(xtrain[i, 1:7]))
      scatter!(t,pred[1:3,:]',label="",color=reds[i])
      display(plot(pl))
    end
    # pl = scatter(t,ode_data[1:3,:]',label="data",color=:blue)
    # scatter!(t,cur_pred[1:3,:]',label="prediction",color=:orange)
    # display(plot(pl))
  end
  global iter += 1
end

# Display the ODE with the initial parameter values.
cb()


Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

# pl = scatter(t,ode_data[1:3,:]',label=["data x" "data y" "data psi"])
# cur_pred = predict_n_ode()