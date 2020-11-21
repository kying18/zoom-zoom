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

nn = FastChain(
    FastDense(1, 32, tanh),
    FastDense(32, 32, tanh),
    FastDense(1, 1)
)

