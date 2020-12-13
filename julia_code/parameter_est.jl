cd(@__DIR__)
# Depends on generate_data.jl which depends on bicycle_model.jl
include("generate_data.jl")
using Plots

#Alternative method for data generation

#=

function example_g(t)

    [sin(t), t, cos(t)]

end


u0 = [1,2,1,3,1,1,1,1]
p = [1292.2, 3,1.006,1.534,0.5,8,0.5,0.1,0.2,0.1]
data = gen_data(u0,example_g,p)
normal_noisy_data = gen_data(u0,example_g,p,add_noise=percent_noise_normal)
normal_noisy_data = gen_data(u0,example_g,p,add_noise=percent_noise_flat)
normal_noisy_data = gen_data(u0,example_g,p,gen_params=change_params_percent_normal)
normal_noisy_data = gen_data(u0,example_g,p,add_noise=percent_noise_normal,gen_params=change_params_percent_normal)
=#

"""
    change_params_percent_normal(params, i, percent_dp)

    Takes params, an array of parameters, an index i, and a floating point
    percentage, percent_dp. For each of the parameters p, it computes a normally
    distributed change dp = N(0.0,percent_dp*p), and decides the sign of the
    change based on the sign of the previous dp. There is a 10% chance that the
    sign of dp differs from the previous dp.
"""
function change_params_percent_normal(params, i, percent_dp)
    # Create a vector to store the new parameters in
    new_params = Vector{Float64}(undef,10)
    # Calculate the previous change in parameters to figure out its sign later
    if i > 2
        prev_dp = params[i-1,:] - params[i-2,:]
    else
        # Set prev_dp to an array of -1.0s for the first parameter change
        prev_dp = -1.0 * ones(Float64,length(params[i-1,:]))
    end
    # Iterate through the parametrs and calculate the next parameter
    for j in 1:length(params[i-1,:])
        # Pick a new parameter
        dp = abs(rand(Normal(0.0, percent_dp[j]*params[i-1,j])))
        # Generate a random number [0,1) to decide the sign of dp
        change_sign = rand(Float64)
        # Figure out the sign of the previous dp
        if prev_dp[j] < 0.0
            sign = -1.0
        else
            sign = 1.0
        end
        # 10% chance of changing the sign of dp from that of the previous dp
        if change_sign <= 0.1
            sign = sign*-1.0
        end
        # Add dp to the previous parameters
        new_params[j] = params[i-1,j] + sign * dp
    end
    return new_params
end

function gen_data(u0, g, p, collect_at=0:1:10; gen_params=false,
    percent_dp=1e-4*ones(Float64,10), add_noise=false, percent_noise=0.01)
    # Figure out number of data points to collect
    len=length(collect_at)
    # Array to store the state data in
    xs=Array{Float64,2}(undef,len,8)
    # Save the initial state
    xs[1,:]=u0
    # Arrays to store dxs and inputs in
    dxs=Array{Float64,2}(undef, len,8)
    inputs=Array{Float64,2}(undef, len,3)

    # Create storage array if parameters are being modified
    if gen_params != false
        # Create an array to store parameters in
        params=Array{Float64,2}(undef,len, length(p))
        # Save the initial parameters
        params[1,:] = p
    end

    # Generate the data
    for itr in 2:len
        # Generate inputs
        inputs[itr-1,:]=g(collect_at[itr-1])

        # If parameters are to be changed,
        if gen_params != false
            # Update p
            p = gen_params(params, itr, percent_dp)
            # Save to params array
            params[itr,:] = p
        end

        # Calculate dx
        dxs[itr,:]=bicycle_model(xs[itr-1,:], inputs[itr-1,:], p)
        # Compute next x
        # xs[itr,:].=xs[itr-1].+dxs[itr]
        xs[itr,:]=xs[itr-1,:].+dxs[itr,:]

        # If noise is to be added
        if add_noise != false
            # Compute next x with noise
            xs[itr,:] = add_noise(xs[itr,:], percent_noise)
        end
    end

    if gen_params == false
        return [collect_at, xs, dxs, inputs]
    else
        return [collect_at, xs, dxs, inputs, params]
    end
end

#=
For parameter estimation, we want to look at the evolution of u over time, with
changing commands and some evolving parameters.
=#

#=
upc= [x = global x value, init ~ 0
    y = global y value, init ~ 0
    psi = global car aligment yaw value, init ~ 0
    vx = forward velocity of the car along its axis in global refrence frame, init ~ 5
    vy = sideways velocity of the car along its axis in global refrence frame, init ~ 0
    r = global yaw rate (dpsi/dt), init 0
    steer = car steering angle relative to its axis, init ~ 0
    Iz = yaw moment of inertia, init ~ 550
    cornering_stiff = cornering stiffness, ranges from [20,000-50,000]
    cla = downforce for velocity ~ - 0.5
    D = command accel
    delta = command steer_rate
    ]
=#

#=
p_stat= [
    m = mass ~ 350kg
    l = car length ~ 3m
    lf = forward dist from center of mass ~ 1.5m
    lr = rear dist from  center of mass ~ 1.5m
    sample_fz = sample downward force at observed cornering stiffness ~ 3430
    rho = density of air ~ 1.2
    g = gravity ~ 9.8 m/s/s
    ]
=#

function bicycle_model_p!(dupc, upc, p_stat, p_command)
    # static params of the car
    m, l, lf, lr, sample_fz, rho, g = p_stat

    # unpack variables
    # states
    u = upc[1:7]
    x, y, psi, vx, vy, r, steer = u
    # dynamic parameters of the car
    Iz, cornering_stiff, cla = p_command[1]
    upc[8] = Iz
    upc[9] = cornering_stiff
    upc[10] = cla
    p = [m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g]
    # commands
    command = p_command[2]
    D, delta = command  # accel command, commanded steer rate
    upc[11] = D
    upc[12] = delta

    # estimate normal
    FzF, FzR = normal_force(u, command, p)

    # compute slip angles
    alpha_f = atan((vy + r * lf) / vx) + steer
    alpha_r = atan((vy - r * lr) / vx)


    # compute tire forces
    F_yf = tire_force(alpha_f, FzF, p)
    F_yr = tire_force(alpha_r, FzR, p)

    # torque to force
    F_net = m * D

    # torque vectoring
    F_xf = lf / l * F_net
    F_xr = lr / l * F_net

    # accel
    ax = 1/m * (F_xr + F_xf * cos(steer) + F_yf * sin(steer)) + r * vy
    ay = 1/m * (F_yr - F_xf * sin(steer) + F_yf * cos(steer)) - r * vx
    a_yaw = 1/Iz * (-lf * F_xf * sin(steer) + lf * F_yf * cos(steer) - lr * F_yr)

    # bicycle model
    dupc[1] = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    dupc[2] = vx * sin(psi) + vy * cos(psi)
    dupc[3] = r
    dupc[4] = ax
    dupc[5] = ay
    dupc[6] = a_yaw
    dupc[7] = delta

end

function bicycle_model_p_spikes!(dupc, upc, p_stat, updated_p)
    # static params of the car
    m, l, lf, lr, sample_fz, rho, g = p_stat

    # unpack variables
    # states
    u = upc[1:7]
    x, y, psi, vx, vy, r, steer = u
    # dynamic parameters of the car
    Iz, cornering_stiff, cla = updated_p
    upc[8] = Iz
    upc[9] = cornering_stiff
    upc[10] = cla
    p = [m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g]
    # commands
    # accel command, commanded steer rate
    D = upc[11]
    delta = upc[12]
    command = D, delta

    # estimate normal
    FzF, FzR = normal_force(u, command, p)

    # compute slip angles
    alpha_f = atan((vy + r * lf) / vx) + steer
    alpha_r = atan((vy - r * lr) / vx)


    # compute tire forces
    F_yf = tire_force(alpha_f, FzF, p)
    F_yr = tire_force(alpha_r, FzR, p)

    # torque to force
    F_net = m * D

    # torque vectoring
    F_xf = lf / l * F_net
    F_xr = lr / l * F_net

    # accel
    ax = 1/m * (F_xr + F_xf * cos(steer) + F_yf * sin(steer)) + r * vy
    ay = 1/m * (F_yr - F_xf * sin(steer) + F_yf * cos(steer)) - r * vx
    a_yaw = 1/Iz * (-lf * F_xf * sin(steer) + lf * F_yf * cos(steer) - lr * F_yr)

    # bicycle model
    dupc[1] = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    dupc[2] = vx * sin(psi) + vy * cos(psi)
    dupc[3] = r
    dupc[4] = ax
    dupc[5] = ay
    dupc[6] = a_yaw
    dupc[7] = delta

end


"""
    update_Iz(Iz0, Iz_rate,t)

A sigmoid function to evolve Iz, the yaw moment of inertia over time. It takes
in Iz0, a floating point number representing the initial yaw moment of inertia,
Iz_rate, a floating point number representing how quickly Iz should evolve,
and t, a floating point number representing the time.
Based on these values, it returns the value of Iz at time t.
"""
function update_Iz(Iz0, Iz_max, Iz_rate, t)
        Iz = Iz0 + (Iz_max-Iz0)*(1.0/(1+Base.MathConstants.ℯ^(-t*Iz_rate))-0.5)
end

test_tspan = range(0, 450.0, length = 901)
plot(test_tspan,
    update_Iz.(550.0, 600.0, 0.01, test_tspan),
    title = "Evolution of Iz Over 5 Periods",
    xlabel = "t",
    ylabel = "Iz",
    legend = false)


"""
    function update_cornering_stiff(cs0, cs_lin, period, t)

A sine + linear function to evolve cornering_stiff over time. It takes in
    cs0 - a floating point number representing the initial cornering stiffness
    cs_lin - a floating point number controlling the linear evolution
    cs_sine - a floating point number controlling the sine amplitude
    period - a floating point number representing the period for the sine
    t - a floating point number representing the time.
Based on these values, it calculates the cornering stiffness as
``
    cornering\_stiffness = cs0\Big(1.0 + cs\_sine*sin\Big(\frac{2\pi t}{period}\Big)+ cs\_lin*t\Big)
``
returns the value of cornering_stiff at time t.
"""
function update_cornering_stiff(cs0, cs_lin, cs_sine, period, t)
    cornering_stiff = cs0 *(1.0 + cs_sine*sin(2.0*pi*t/period) + cs_lin * t)
end

test_tspan = range(0, 450.0, length = 901)
plot(test_tspan,
    update_cornering_stiff.(20000.0, 0.002, 0.1, 90, test_tspan),
    title = "Evolution of Cornering Stiffness Over 5 Periods",
    xlabel = "t",
    ylabel = "Cornering Stiffness",
    legend = false)

"""
    update_cla(d_cla, period, t)

A sine function to evolve cla over time. It takes in
    d_cla - a floating
    period - a floating point number representing the period for the sine
    t - a floating point number representing the time.
Based on these values, it returns the value of cla at time t.
"""
function update_cla(d_cla, period, t)
    cla = d_cla*sin(2.0*pi*t/period) - 0.5
end

test_tspan = range(0, 90.0, length = 181)
plot(test_tspan,
    update_cla.(0.2, 90.0, test_tspan),
    title = "Evolution of Cla Over 1 Period",
    xlabel = "t",
    ylabel = "Cla")

function update_p(p_update, t)
    new_p = Vector{Float64}(undef,3)
    Iz0, Iz_max, Iz_rate, cs0, cs_lin, cs_sine, period, d_cla = p_update
    new_p[1] = update_Iz(Iz0, Iz_max, Iz_rate, t)
    new_p[2] = update_cornering_stiff(cs0, cs_lin, cs_sine, period, t)
    new_p[3] = update_cla(d_cla, period, t)
    return new_p
end

function com(t)
    [5,0.1] .* 2 .*(rand(Float64, (2)).-0.5)
end

function update_commands(t)
    if isapprox(t % 2.0, 0.0,
end


isapprox(102.001 % 2.0, 0, atol=1e-1)

function diffeq_bicycle_model_p!(dupc, upc, p_diffeq, t)
    # Unpack the static parameters and the
    p_stat, p_update = p_diffeq
    updated_p = update_p(p_update, t)
    p_command = [updated_p, next!(command_s)]
    bicycle_model_p!(dupc, upc, p_stat, p_command)
end

function diffeq_bicycle_model_p_spikes!(dupc, upc, p_diffeq, t)
    # Unpack the static parameters and the
    p_stat, p_update = p_diffeq
    updated_p = update_p(p_update, t)
    bicycle_model_p_spikes!(dupc, upc, p_stat, updated_p)
end

# Set up Sobol sequence for generating commands
minmax_delta = 30.0*pi/360.0 # max change in delta is 15 degrees
command_lb = [-20.0, -minmax_delta]
command_ub = [15.0, minmax_delta]
global command_s = SobolSeq(command_lb, command_ub)

# Define the static parameters
m = 350.0
l = 3.0
lf = 1.5
lr = 1.5
sample_fz = 3430.0
rho = 1.2
g = 9.8
# Assemble p_stat
p_stat = [m, l, lf, lr, sample_fz, rho, g]
# Define the parameters for updating the parameters
Iz0 = 550.0
Iz_max = 600.0
Iz_rate = 0.01
cs0 = 20000.0
cs_lin = 0.002
cs_sine = 0.1
period = 90.0
d_cla = 0.2
cla0 = -0.5
# Assemble p_update
p_update = [Iz0, Iz_max, Iz_rate, cs0, cs_lin, cs_sine, period, d_cla]
# Assemble p_diffeq for the ODEproblem
p_diffeq = [p_stat, p_update]
# Define the initial upc vector
# [x, y, phi, vx, vy, r, steer, Iz, cornering_stiff, cla, D, delta]
upc0 = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, Iz0, cs0, cla0, 0.0, 0.0]

test_dupc = zeros(12)
test_upc = ones(12)
test_p_command = [[3.0,4,5],[6,7]]
bicycle_model_p!(test_dupc, test_upc, p_stat, test_p_command)
diffeq_bicycle_model_p!(test_dupc, upc0, p_diffeq, 0.0)
diffeq_bicycle_model_p_spikes!(test_dupc, upc0, p_diffeq, 0.0)

tspan = (0.0, 30.0)
prob=ODEProblem(diffeq_bicycle_model_p!, upc0, tspan, p_diffeq) #doesn't set dt and uses adaptive time steping
sol = solve(prob, maxiters = 1e6, saveat=0.1)

#plot path of car
plot(sol, vars=(1,2)) # (x, y)
plot(sol, vars=8) # Iz
plot(sol, vars=9) # Cornering stiffness
plot(sol, vars=10) # Cornering stiffness

test_dupc
test_upc


spike_tmax = 150.0
spike_tspan = (0.0,spike_tmax)
dosetimes = 0.0:0.5:spike_tmax
affect!(integrator) = integrator.u[11:12] .= next!(command_s)
cb = PresetTimeCallback(dosetimes,affect!)
spike_prob=ODEProblem(diffeq_bicycle_model_p_spikes!, upc0, tspan, p_diffeq)
spike_sol = solve(spike_prob,callback=cb, max_iter=1e7, saveat=0.05)

#plot path of car
plot(spike_sol, vars=(1,2), xlabel="x", ylabel="y") # (x, y)
plot(spike_sol, vars=8, xlabel="t", ylabel="Iz", legend=false) # Iz
plot(spike_sol, vars=9, xlabel="t", ylabel="Cornering Stiffness", legend=false) # Cornering stiffness
plot(spike_sol, vars=10, xlabel="t", ylabel="Cla", legend=false)
plot(spike_sol, vars=11, xlabel="t", ylabel="D", legend=false)
plot(spike_sol, vars=12, xlabel="t", ylabel="delta", legend=false)
