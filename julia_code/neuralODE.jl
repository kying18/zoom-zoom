cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Flux, DiffEqBase, OrdinaryDiffEq, Plots, DataStructures, DiffEqFlux, Zygote

U0=Float32[1.0,1.0,1.0,5.0,1.0,1.0,1.0,1.0,1.0] #9
PT=Float32[350.0,3.0,1.5,1.5,550.0,10000.0,3430.0,1.2,-0.5,9.8] #10
P=PT./[100,1,1,1,100,10000,1000,1,1,1]
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

n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)

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
cb = function () #callback function to observe training
  display(loss_n_ode())
  # plot current prediction against data
  cur_pred = predict_n_ode()
  pl = scatter(t,ode_data[1:3,:]',label="data",color=:blue)
  scatter!(t,cur_pred[1:3,:]',label="prediction",color=:orange)
  display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
