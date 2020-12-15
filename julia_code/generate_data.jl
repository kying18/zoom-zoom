cd(@__DIR__)
include("bicycle_model.jl")
using Random
using Distributions
using DifferentialEquations
using Sobol
using DiffEqCallbacks

#=
#Check that this works vv

tspan = (0.0, 20.0)
prob=ODEProblem(diffeq_bicycle_model,u0,tspan,p) #doesn't set dt and uses adaptive time steping
sol = solve(prob)

#plot path of car
plot(sol, vars=(1,2))
=#

#=
ODE problem example
=#
#=
#collect data
tspan = (0.0, 0.5)

data=[]

function com(t)
    [5,0.1] .* 2 .*(rand(Float64, (2)).-0.5)
end

for i in 1:10
    println(i)
    function com(t)
        [5,0.1] .* 2 .*(rand(Float64, (2)).-0.5)
    end
    p=[350,3,1.5,1.5,550,10000,3500,1.2,-0.5,9.8].* 2 .*(rand(Float64, (length(p_ex))).-0.5) #wrong, some of the params are linked to eachother & g=9.8
    u0=[0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0].+ 2 .*(rand(Float64, (length(u_ex))).-0.5)

    prob=ODEProblem(diffeq_bicycle_model,u0,tspan,p,dt=0.001,saveat=0.1)
    sol = solve(prob)
    push!(data,[p,u0,sol.t, sol.u])
end
=#

#=
Random data generation
    varied states: psi, vx, vy, r, steer
    varied commands: D (acceleration), delta (steering rate)
    varied parameters: cornering_stiff
    integration with Dormand Prince method
    suggested step size = 0.001
=#

#=
Dormand Prince Method
=#
# Define the coefficients for the Dormand Prince method
c = [0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0]
b = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
a2 = 1.0/5.0
a3 = [3.0/40.0, 9.0/40.0]
a4 = [44.0/45.0, -56.0/15.0, 32.0/9.0]
a5 = [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0]
a6 = [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0]
a7 = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0]

# A function that calculates one step using the Dormand Prince method.
function dormandprince(f, u, p, command, dt)
    # Calculate the stages
    k1 = f(u,p,command)
    k2 = f(muladd(dt,a2*k1,u),p,command)
    k3 = f(muladd(dt,muladd(a3[1],k1,a3[2]*k2),u),p,command)
    k4 = f(muladd(dt,muladd(a4[1],k1,muladd(a4[2],k2,a4[3]*k3)),u),p,command)
    k5 = f(muladd(dt,muladd(a5[1],k1,muladd(a5[2],k2,muladd(a5[3],k3,a5[4]*k4))),u),p,command)
    k6 = f(muladd(dt,muladd(a6[1],k1,muladd(a6[2],k2,muladd(a6[3],k3,muladd(a6[4],k4,a6[5]*k5)))),u),p,command)
    # The last step is only needed for error estimation
    # Calculate the next u
    u_next = muladd(dt,muladd(b[1],k1,muladd(b[3],k3,muladd(b[4],k4,muladd(b[5],k5,b[6]*k6)))),u)
    # Return the next u
    out=u_next
end

#=
Noise functions
=#
"""
    flat_noise(u, percent_noise)

Computes noise by adding or subtracting a specified percentage of the provided
state array as error: noisy_u = u +/- (percent)*u
Takes in u, a state vector, and percent_noise, a float. Outputs noisy_u, the
state vector with added noise.
"""
function flat_noise(u, percent_noise)
    # Create array of +/-1 the size of u
    noise = rand([Float64(-1.0),Float64(1.0)],length(u))
    # Multiply the percent noise by the u values, and then multiply by the
    # +/- noise array. Add to the u values.
    noisy_u = u + percent_noise * u .* noise
    return noisy_u
end

"""
    normal_noise(u, percent_noise)

Adds normally distributed noise by calculating noise N(0.0, percent_noise) and
computing noisy_u = u + N(0.0, percent_noise)*u.
Takes in u, a state vector, and percent_noise, a float. Outputs noisy_u, the
state vector with added noise.
"""
function normal_noise(u, percent_noise)
    # Create an array the size of u
    noisy_u = Vector{Float64}(undef,length(u))
    # For each u, sample from the N(u, percent*u) distribution
    noisy_u .= u .* (Float64(1.0) .+ rand(Normal(0.0,percent_noise), length(u)))
    return noisy_u
end

#=
Unaccounted functions
=#

"""
    ex_unacc_sine(u, p, unacc_p)
This is an example function to add some behavior unaccounted for in the bicycle
model to the data. It takes in a state vector u, a vector for parameters for the
bicycle model p, and a vector for extra parameters for the function, unacc_p.
In this case unacc_p = [percent]
This function will add percent*sin(u) to u.
    unacc_u = u + u*percent*sin(u)
"""
function ex_unacc_sine(u,p,unacc_p)
    # Create an array the size of u
    unacc_u = Vector{Float64}(undef,length(u))
    # For each u, sample from the N(u, percent*u) distribution
    unacc_u = u + unacc_p[1] * u .* sin.(u)
    return unacc_u
end

#=
Data generator w/ "random" initial states/commands/cornering_stiff
=#

"""
gen_data(N_data, lb, ub; dt = 0.001, add_noise=false,
    percent_noise=0.01, add_unacc=false, unacc_p=false)
gen_data takes in
    N_data = a number of data points
    lb = a vector of lower bounds for the varied values, and
    ub = a vector of upper bounds for the varied values.
It generates initial states u with [x, y] = [0,0], commands, and parameters,
then uses them with the Dormand Prince method to calculate the next state u_next.

The default returns an array with rows of:
    [varied u, commands, varied p, u_next]
If add_noise is used, returns an array with rows of:
    [varied u, commands, varied p, u_next, noisy_u]
If add_unacc is used, returns an array with rows of:
    [varied u, commands, varied p, u_next, unacc_u]
If add_noise and add_unacc are used, returns an array with rows of:
    [varied u, commands, varied p, u_next, noisy_u, unacc_u, noisy_unacc_u]

Optional args:
    dt - A float. Changes the time step the Dormand Prince method uses.
         Defaults to 0.001
    add_noise - a function add_noise(u,percent_noise) to add noise to u_next.
                Default is false.
    percent_noise - a float that controls how much noise is added.
                    Default is 0.01 (1%).
    add_unacc - a function add_unacc(u,p,unacc_p) to add unaccounted behavior to u_next.
               Default is false.
    unacc_p - a vector of parameters to use with add_unacc.
              Default is empty vector [].
"""
function gen_data(N_data, lb, ub; dt = 0.001, add_noise=false,
    percent_noise=0.01, add_unacc=false, unacc_p=Vector{Float64}[])
    # Set up Sobol sequence generator to vary the following states, commands,
    # and parameters.

    s = SobolSeq(lb, ub)
    # Skip the inital portion of the LDS
    skip(s, N_data)

    # Set the fixed parameter values
    rho = 1.2    # air density
    g = 9.8      # gravity

    # Initialize an array to store data
    # Figure out how many parameters are changing
    N_dp = Int(length(lb)-7)

    # Collect vanilla states only
    if (add_noise == false) & (add_unacc == false)
        data = Array{Float64}(undef,(N_data, Int(length(lb)+7)))
    # Collect vanilla states and either states + noise or states + extra
    elseif ((add_noise == false) & (add_unacc != false)) |
        ((add_noise != false) & (add_unacc == false) )
        data = Array{Float64}(undef,(N_data, Int(length(lb)+14)))
    # Collect vanilla states, states + noise, states + extra, and
    # states + noise + extra
    else
        data = Array{Float64}(undef,(N_data, Int(length(lb)+28)))
    end

    # Generate the data
    for i in 1:N_data
        # Generate the new state, command, and parameter values
        x = next!(s)
        # Update the states, commands, and changed parameters
        u = vcat(zeros(2), x[1:5])
        commands = x[6:7]
        m = x[8]     # mass
        l = x[9]     # length
        lflr = x[10] # lf/lr
        Iz = x[11]   # Iz
        cornering_stiff = x[12]    # cornering stiffness
        sample_fz = x[8]*g # sample_fz = mass * gravity
        cla = x[13]  # cla

        # Assemble parameter vector
        p = [m, l, lflr, lflr, Iz, cornering_stiff, sample_fz, rho, cla, g]

        # Calculate the next u with the Dormand Prince method
        u_next = dormandprince(bicycle_model, u, p, commands, dt)

        # Add noise
        if add_noise != false
            # Compute next x with noise
            noisy_u = add_noise(u_next, percent_noise)
        end
        # Add extra
        if add_unacc != false
            unacc_u = add_unacc(u_next, p, unacc_p)
        end
        # Add noise and extra
        if (add_noise != false) & (add_unacc != false)
            noisy_unacc_u = add_unacc(noisy_u, p, unacc_p)
        end

        # Save data
        if (add_noise == false) & (add_unacc == false)
            data[i,:] = vcat(x, u_next)
        elseif (add_noise != false) & (add_unacc == false)
            data[i,:] = vcat(x, u_next, noisy_u)
        elseif (add_noise == false) & (add_unacc != false)
            data[i,:] = vcat(x, u_next, unacc_u)
        else
            data[i,:] = vcat(x, u_next, noisy_u, unacc_u, noisy_unacc_u)
        end
    end
    return data
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

#=
gen_data(5, lb, ub)
# output row: [psi0, vx0, vy0, r0, steer0, D, delta, cornering_stiff, x, y, psi, vx, vy, r, steer]
gen_data(5, lb, ub, add_noise=normal_noise)
# output row: [psi0, vx0, vy0, r0, steer0, D, delta, cornering_stiff, x, y, psi, vx, vy, r, steer, x_noise, y_noise, psi_noise, vx_noise, vy_noise, r_noise, steer_noise]
gen_data(5, lb, ub, add_unacc=ex_unacc_sine, unacc_p = [0.04])
gen_data(5, lb, ub, add_noise=normal_noise, add_unacc=ex_unacc_sine, unacc_p = [0.04])
=#

#=
Plotting example with commands at fixed time step
=#

#=
# Set initial values for example
u0_ex=[0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0]

# Set up Sobol sequence for generating commands
minmax_delta = 30.0*pi/360.0 # max change in delta is 15 degrees
command_lb = [min_D, -minmax_delta] # Change value above to change bounds
command_ub = [max_D, minmax_delta]
global command_s = SobolSeq(command_lb, command_ub)

callback_tmax = 50.0
callback_tspan = (0.0,callback_tmax)
dosetimes = 0.0:0.5:callback_tmax
affect!(integrator) = integrator.u[8:9] .= next!(command_s)
cb = PresetTimeCallback(dosetimes,affect!)
callback_prob=ODEProblem(bicycle_model_callback!, u0_ex, callback_tspan, p_ex)
callback_sol = solve(callback_prob,callback=cb, saveat=0.05)

plot(callback_sol, vars=(1,2), xlabel="x", ylabel="y", title="Vehicle Trajectory") # (x, y)
=#
