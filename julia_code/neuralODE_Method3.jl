cd(@__DIR__)
using Pkg
Pkg.activate(".")

using DiffEqFlux, Flux, DiffEqBase, OrdinaryDiffEq, Plots, DataStructures, Zygote
include("./pack_race_data.jl")

start = 300
num_dp = 200

num_long=200

tspan = (577.596f0,805.436f0)
tstart = tspan[1]+0.04*start
t = tstart:0.04:tstart+num_dp*0.04#tspan[2]
t_long = tstart+num_dp*0.04:0.04:tstart+num_dp*0.04+(num_dp)*0.04#tspan[2]
collect(t_long)
#=
batch_size=50
dp_ps=zeros(10)
dp_u0s=zeros(9)
dp_sols=zeros(9,16)

dp_ps=zeros(10,batch_size)
dp_u0s=zeros(9,batch_size)
l = length(collect(t))
dp_sols=zeros(9,16,batch_size)
function newbatch()
  for i in batch_size
    global dp=gen_rand_time_data(t)
    global dp_ps[:,i] .= dp[1]
    global dp_sols[:,:,i] .= dp[2]
    global dp_u0s .= dp[2][:,1]

  end
end

newbatch()
=#


global dp=[gg[1][:,2] gg[2][:,2] gg[3][:,2] gg[4][:,2] gg[5][:,2] gg[6][:,2] gg[7][:,2] gg[8][:,2] gg[9][:,2]][1:end-1,:]'
dp_u0=dp[1:7,1]


dudt = Chain(x->vcat(x[3:end],dp_p),
             Dense(17,50,tanh),
             Dense(50,30,tanh),
             Dense(30,7))

nodeitr=0
function nodeaffect!(integrator)
  nodeitr+=1
    dp_p[11:12]=dp[8:9,start+nodeitr]
end
span = (t[1],t[end])
ts = t
node_cb = PresetTimeCallback(t,nodeaffect!)
n_ode = NeuralODE(dudt,span,Tsit5(),saveat=ts,reltol=1e-4,abstol=1e-4, cb=node_cb) # decrease tolerances
ps = Flux.params(n_ode)



#TODO: Loss for each starting u0/parameter

function predict_n_ode()
  global nodeitr = 0
  n_ode(dp_u0)
end



function loss_n_ode()
  global dp_u0=dp[1:7,start]

  sum(abs2,dp[1:7,start:start+num_dp-1].- Array(predict_n_ode())[:,1:num_dp])
  #end
end
#@btime loss_n_ode()
iter = 0
cb = function () #callback function to observe training
  global start += 1
  if iter % 20 == 0
    #global start+=1

    cur_loss =loss_n_ode()
    println("itr: ",iter," Loss: ",cur_loss)
  end
  if iter % 200 == 0

    # plot current prediction against data
    curpred = predict_n_ode()
    pl=plot(t_long,dp[1:2,start+num_dp:start+num_dp+num_long]',color=["blue" "skyblue"], labels = ["" ""])
    global dp_u0=curpred[1:7,end]
    longpred = predict_n_ode()
    plot!(t_long,longpred[1:2,:]',color=["red" "orange"], labels = ["" ""])
    scatter!(t,dp[1:2,start:start+num_dp-1]',color=["blue" "skyblue"], labels = ["true x" "true y" ])
    scatter!(t,curpred[1:2,:]',color=["red" "orange"], labels = ["predicted x" "predicted y" ], xlabel = "t (s)", ylabel = "distance (m)", title=string("After, itr: ",iter," Loss: ", round(cur_loss/(batch_size*15),digits=3)))


    display(plot(pl))


  end
  global iter += 1
end



#TODO: Train
data = Iterators.repeated((), 1000)
opt = ADAM(0.02)
cb()
Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
