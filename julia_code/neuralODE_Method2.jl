cd(@__DIR__)
using Pkg
Pkg.activate(".")

using DiffEqFlux, Flux, DiffEqBase, OrdinaryDiffEq, Plots, DataStructures, Zygote
include("./generate_data.jl")

datasize = 5
tspan = (0.0f0,1.0f0)
t = tspan[1]:0.2:tspan[2]


batch_size=10

l = length(collect(t))



dp_p=zeros(12)
dp_u0=zeros(7)
dp_sol=zeros(9,l)

dp_ps=zeros(12,batch_size)
dp_u0s=zeros(7,batch_size)
dp_sols=zeros(9,length(collect(t)),batch_size)
function newbatch()
  for i in 1:batch_size
    global dp=gen_rand_time_data_mod(t)
    global dp_ps[:,i] .= vcat(dp[1],[0,0])
    global dp_sols[:,:,i] .= dp[2]
    global dp_u0s .= dp[2][1:7,1]

  end
end

newbatch()


dudt = Chain(x->vcat(x[3:end],dp_p),
             Dense(17,50,tanh),
             Dense(50,30,tanh),
             Dense(30,7))

nodeitr=0
function nodeaffect!(integrator)
  nodeitr+=1
    dp_p[11:12]=dp_sol[8:9,nodeitr]
end
node_cb = PresetTimeCallback(t,nodeaffect!)
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t,reltol=1e-4,abstol=1e-4, cb=node_cb) # decrease tolerances
ps = Flux.params(n_ode)


#TODO: Loss for each starting u0/parameter

function predict_n_ode()
  nodeitr=0
  n_ode(dp_u0)
end

function loss_n_ode()
  loss = 0
  for i in 1:batch_size
    global dp_p = dp_ps[:,i]
    global dp_sol= dp_sols[:,:,i]
    global dp_u0=dp_u0s[:,i]
    loss+=sum(abs2,dp_sols[1:7,:,i].- Array(predict_n_ode())[1:7,:])
  end
  loss/batch_size
end

ts = 0

dt = 0
iter = 0
cb = function () #callback function to observe training
  if(iter>0)
    global dt+=time()-ts
  end
  if iter % 200 == 0
    cur_loss=loss_n_ode()
    println("itr: ",iter," Loss: ",cur_loss)
    # plot current prediction against data
    pl = scatter(t,dp_sol[1:2,:]',color=["blue" "skyblue"], labels = ["true x" "true y" ])
    scatter!(t,predict_n_ode()[1:2,:]',color=["red" "orange"], labels = ["predicted x" "predicted y" ], xlabel = "t (s)", ylabel = "distance (m)", title=string("After, itr: ",iter," Loss: ", round(cur_loss/(batch_size*15),digits=3)))
    display(plot(pl))
    savefig(string("NeuralODE2 itr ",iter))
    newbatch()
    cur_loss=loss_n_ode()
    println("(new batch) itr: ",iter," Loss: ",cur_loss)
    pl = scatter(t,dp_sol[1:2,:]',color=["blue" "skyblue"], labels = ["true x" "true y" ])
    scatter!(t,predict_n_ode()[1:2,:]',color=["red" "orange"], labels = ["predicted x" "predicted y" ], xlabel = "t (s)", ylabel = "distance (m)", title=string("Before, itr: ",iter," Loss: ", round(cur_loss/(batch_size*15),digits=3)))
    display(plot(pl))
    savefig(string("NeuralODE2 itr ",iter, " nb"))


elseif iter % 10 == 0
  cur_loss=loss_n_ode()
  println("itr: ",iter," Loss: ",cur_loss)
  # plot current prediction against data
  pl = scatter(t,dp_sol[1:2,:]',color=["blue" "skyblue"], labels = ["true x" "true y" ])
  scatter!(t,predict_n_ode()[1:2,:]',color=["red" "orange"], labels = ["predicted x" "predicted y" ], xlabel = "t (s)", ylabel = "distance (m)", title=string("After, itr: ",iter," Loss: ", cur_loss/batch_size))
  display(plot(pl))
  display(ps)
end

  global iter += 1
  global ts=time()
end



#TODO: Train
data = Iterators.repeated((), 4000)
opt = ADAM(0.002)#Nesterov(0.0001, 0.50)
cb()
Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

println("total time: "dt/iter)
