cd(@__DIR__)
# Depends on generate_data.jl which depends on bicycle_model.jl
include("generate_data.jl")
using Plots
using BenchmarkTools

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

    # dynamic parameters of the car
    Iz, cornering_stiff, cla = p_command[1]
    upc[8] = Iz
    upc[9] = cornering_stiff
    upc[10] = cla
    p = [m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g]
    # commands
    command = p_command[2]
    upc[11:12] .= command

    # Calculate derivative of states
    dupc[1:7] .= bicycle_model(u, p, command)

end

function bicycle_model_p_spikes!(dupc, upc, p_diffeq, t)
    # Unpack parameter arrays
    p_stat, p_update = p_diffeq

    # static params of the car
    m, l, lf, lr, sample_fz, rho, g = p_stat

    # unpack variables
    # states
    u = upc[1:7]
    # dynamic parameters of the car
    Iz, cornering_stiff, cla = update_p(p_update, t)
    upc[8] = Iz
    upc[9] = cornering_stiff
    upc[10] = cla
    p = [m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g]

    # accel command, commanded steer rate
    command = upc[11:12]

    # Calculate derivative of states
    dupc[1:7] .= bicycle_model(u, p, command)

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

function diffeq_bicycle_model_p!(dupc, upc, p_diffeq, t)
    # Unpack the static parameters and the
    p_stat, p_update = p_diffeq
    updated_p = update_p(p_update, t)
    p_command = [updated_p, next!(command_s)]
    bicycle_model_p!(dupc, upc, p_stat, p_command)
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
global p_stat = [m, l, lf, lr, sample_fz, rho, g]
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

tspan = (0.0, 30.0)
prob=ODEProblem(diffeq_bicycle_model_p!, upc0, tspan, p_diffeq) #doesn't set dt and uses adaptive time steping
sol = solve(prob, maxiters = 1e6, saveat=0.1)

#plot path of car
plot(sol, vars=(1,2)) # (x, y)
plot(sol, vars=8) # Iz
plot(sol, vars=9) # Cornering stiffness
plot(sol, vars=10) # Cornering stiffness


spike_tmax = 100.0
spike_tspan = (0.0,spike_tmax)
dosetimes = 0.0:0.5:spike_tmax
affect!(integrator) = integrator.u[11:12] .= next!(command_s)
cb = PresetTimeCallback(dosetimes,affect!)
spike_prob=ODEProblem(bicycle_model_p_spikes!, upc0, spike_tspan, p_diffeq)
spike_sol = solve(spike_prob,callback=cb, saveat=0.05)

#plot path of car
plot(spike_sol, vars=(1,2), xlabel="x", ylabel="y", title="Vehicle Trajectory") # (x, y)
plot(spike_sol, vars=8, xlabel="t", ylabel="Iz", title="Evolution of Iz", legend=false) # Iz
plot(spike_sol, vars=9, xlabel="t", ylabel="Cornering Stiffness", title="Evolution of Cornering Stiffness", legend=false) # Cornering stiffness
plot(spike_sol, vars=10, xlabel="t", ylabel="Cla", title="Evolution of Cla", legend=false)
plot(spike_sol, vars=11, xlabel="t", ylabel="D", title="Parameter Estimation Input Acceleration", legend=false)
plot(spike_sol, vars=12, xlabel="t", ylabel="delta", title="Parameter Estimation Input Steering", legend=false)


#=
Parameter estimation df/dp calculations
=#
function calc_dF_yfr_dcla(vx, cornering_stiff, alphafr, l, lfr, sample_Fz, rho)
    dF_yfr_dcla = 0.5 * alphafr * cornering_stiff * lf * rho * vx^2.0 / (sample_Fz * l)
    return dF_yfr_dcla
end

function calc_dF_yfr_dcs(alphafr, Fzfr, sample_Fz)
    dF_yfr_dcs = -alphafr * Fzfr / sample_Fz
    return dF_yfr_dcs
end

function calc_d_ayaw_dcscla(Iz, steer, lf, lr, dF_yf_dcscla, dF_yr_dcscla)
    d_ayaw_dcscla = ( lf*cos(steer)*dF_yf_dcscla - lr*dF_yr_dcscla )/Iz
    return d_ayaw_dcscla
end

function calc_d_ayaw_dIz(a_yaw, Iz)
    d_ayaw_dIz = a_yaw/Iz
    return d_ayaw_dIz
end

function calc_d_ay_dcscla(steer, m, dF_yf_dcscla, dF_yr_dcscla)
    d_ay_dcscla = ( cos(steer)*dF_yf_dcscla + dF_yr_dcscla )/m
    return d_ay_dcscla
end

function calc_d_ax_dcscla(steer, m, dF_yf_dcscla)
    d_ax_dcscla = ( sin(steer)*dF_yf_dcscla )/m
    return d_ax_dcscla
end


function bicycle_model_est_p_spikes!(dudpc, udpc, p_stat, t)
    # static params of the car
    m, l, lf, lr, sample_fz, rho, g = p_stat

    # unpack variables
    # states
    u = udpc[1:7]
    # dynamic parameters of the car
    udpc[8] = Iz
    udpc[9] = cornering_stiff
    udpc[10] = cla
    p = [m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g]

    # accel command, commanded steer rate
    command = udpc[11:12]

    # Calculate derivative of states
    dudpc[1:7] .= bicycle_model(u, p, command)

end

function bicycle_model2(u, p, command)
    # Create an array to store du in.
    du = Vector{Float64}(undef, length(u))


    # du[1] = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    # du[2] = vx * sin(psi) + vy * cos(psi)
    # du[3] = r
    # du[4] = ax
    # du[5] = ay
    # du[6] = a_yaw
    # du[7] = delta
    # return du
end

function bicycle_model_est_p_Jacobian(udpc)
    # static params of the car
    m, l, lf, lr, sample_fz, rho, g = p_stat

    # unpack variables
    # states
    u = udpc[1:7]
    # unpack variables
    x, y, psi, vx, vy, r, steer = u
    # dynamic parameters of the car
    Iz = udpc[8]
    cornering_stiff = udpc[9]
    cla = udpc[10]
    p = [m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g]

    # accel command, commanded steer rate
    command = udpc[11:12]
    D, delta = command  # velocity along path, accel command, commanded steer rate

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
    xdot = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    ydot = vx * sin(psi) + vy * cos(psi)
    psidot = r
    vxdot = ax
    vydot = ay
    rdot = a_yaw
    steerdot = delta
    return [xdot, ydot, psidot, vxdot, vydot, rdot, steerdot]

end

u_ex = [0.0, 0.0, pi, 5.0, 2.0, 0.01, -2.0]
dp_ex = [540.0, 20000.0, -0.34]
com_ex = [4.0, 0.02]
udpc_ex = vcat(u_ex, dp_ex, com_ex)
dudpc_ex = zeros(12)

bicycle_model_est_p_Jacobian(udpc_ex)

using ForwardDiff
@btime BM_Jacobian = ForwardDiff.jacobian(bicycle_model_est_p_Jacobian, udpc_ex)

println("wah")
for i in 1:7
    println(BM_Jacobian[i,1:7])
end
