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
using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, Plots, Statistics, DiffEqSensitivity

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

loss() = 0 # TODO replace this

############ L2 Loss based on state params #############
function loss_L2()


end

opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(loss())
  end
end
display(loss())

Flux.train!(loss, Flux.params(nn), data, opt; cb=cb)
