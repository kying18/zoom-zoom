cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux, DiffEqBase, OrdinaryDiffEq, Plots, DataStructures, DiffEqFlux, Zygote
using DiffEqFlux
##########################################################
####### DAVID's OG NEURAL ODE ###########################
##########################################################
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


##########################################################
########## KYLIES ADDITIONS ##############################
##########################################################

# just set the parameters and normalize them
PT=Float32[350.0,3.0,1.5,1.5,550.0,10000.0,3430.0,1.2,-0.5,9.8] #10
P=PT./[100,1,1,1,100,10000,1000,1,1,1] # normalization

function get_data_no_noise(num_data, lb, ub)
  data = gen_data(num_data, lb, ub)
  x = hcat(rand(num_data, 2), data[:,1:7])

  # [psi0, vx0, vy0, r0, steer0, D, delta, cornering_stiff, x, y, psi, vx, vy, r, steer]
  # 1:7 inputs
  # 9:15 outputs

  return x
end

num_time_points = 20
tspan = (0.0f0,1.0f0)
t = range(tspan[1],tspan[2],length=num_time_points)

datasize = 20

num_train = floor(Int, 0.8*datasize)
num_valid = floor(Int, 0.1*datasize)
num_test = floor(Int, 0.1*datasize)

dudt = Chain(x->vcat(x,P),
             Dense(17,30,relu),
             Dense(30,30,relu),
             Dense(30,7))

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-4,abstol=1e-4) # decrease tolerances

ps = Flux.params(n_ode)

xtrain = get_data_no_noise(num_train, lb, ub)
ode_data = Array{Float32, 2}(undef, 9, 0)
for i=1:num_train
  u0 = xtrain[i, :]
  prob = ODEProblem(diffeq_bicycle_model,u0,tspan,P)
  global ode_data = hcat(ode_data, Array(solve(prob,Tsit5(),saveat=t)))
end

pl = scatter() # clears the plot
reds=range(colorant"lightsalmon", stop=colorant"red3", length=3)
blues=range(colorant"skyblue", stop=colorant"darkblue", length=3)
# pred = Array(n_ode(U0[1:7])) # Get the prediction using the correct initial condition
for i=1:num_train
  # scatter!(t,ode_data[1:3,(i-1)*num_time_points+1:i*num_time_points]',label="",color=blues[i])
  scatter!(t,ode_data[1:3,(i-1)*num_time_points+1:i*num_time_points]',label="",color=:blue)
  # scatter!(t,ode_data[1:3,i*num_time_points:(i+1)*num_time_points-1]')
end
scatter!()
function predict_n_ode()
  preds = Array{Float32, 2}(undef, 7, 0)
  for i=1:num_train
    pred = Array(n_ode(xtrain[i, 1:7]))
    # scatter!(t,pred[1:3,:]',label="",color=reds[i])
    preds = hcat(preds, pred)
  end
  return preds
end
loss_n_ode() = sum(abs2,Array(ode_data)[1:7,:].- predict_n_ode())
predict_n_ode()
scatter!()
loss_n_ode()

data = Iterators.repeated((), 1000)
opt = ADAM(0.002)
iter = 0
cb = function () #callback function to observe training
  if iter % 10 == 0
    display(loss_n_ode())
    # plot current prediction against data
    # pl = scatter()
    for i=1:num_train
      pl = scatter(title="True vs Predicted X, Y, Psi")
      scatter!(t,ode_data[1:3,(i-1)*num_time_points+1:i*num_time_points]',label=["true x" "true y" "true psi"],color=blues',legend=:topleft)
      pred = Array(n_ode(xtrain[i, 1:7]))
      scatter!(t,pred[1:3,:]',label=["pred x" "pred y" "pred psi"],color=reds',legend=:topleft)
      xlabel!("Time (s)")
      ylabel!("Value (m or rad)")
      display(plot(pl))
      savefig(pl, string("true_pred_", i))
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

### validation/test
xvalid = get_data_no_noise(num_valid, lb, ub)
ode_data = Array{Float32, 2}(undef, 9, 0)
for i=1:num_valid
  u0 = xvalid[i, :]
  prob = ODEProblem(diffeq_bicycle_model,u0,tspan,P)
  ode_data = hcat(ode_data, Array(solve(prob,Tsit5(),saveat=t)))
end
for i=1:num_valid
  pl = scatter()
  scatter!(t,ode_data[1:3,(i-1)*num_time_points+1:i*num_time_points]',label="",color=blues[i])
  pred = Array(n_ode(xvalid[i, 1:7]))
  scatter!(t,pred[1:3,:]',label="",color=reds[i])
  display(plot(pl))
  savefig(pl, string("valid_", i))
end
function predict_n_ode_valid()
  preds = Array{Float32, 2}(undef, 7, 0)
  for i=1:num_valid
    pred = Array(n_ode(xvalid[i, 1:7]))
    # scatter!(t,pred[1:3,:]',label="",color=reds[i])
    preds = hcat(preds, pred)
  end
  return preds
end
loss_n_ode() = sum(abs2,Array(ode_data)[1:7,:].- predict_n_ode_valid())
print(loss_n_ode())
# 2 layer (50) 5000 iters 0.002 lr 16566.21937421601 loss
