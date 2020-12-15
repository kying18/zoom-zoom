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
function calc_dFyfr_dcla(vx, cornering_stiff, alphafr, l, lfr, sample_Fz, rho)
    dFyfr_dcla = 0.5 * alphafr * cornering_stiff * lf * rho * vx^2.0 / (sample_Fz * l)
    return dFyfr_dcla
end

function calc_dFyfr_dcs(alphafr, Fzfr, sample_Fz)
    dFyfr_dcs = -alphafr * Fzfr / sample_Fz
    return dFyfr_dcs
end

#=
Parameter estimation df/du calculations
=#
function calc_dalphafr_dvy(vx, vy, r, lfr)
    dalphafr_dvy = vx / ( (vy + r*lfr)^2.0 +vx^2.0 )
end

function calc_dalphafr_dvx(vx, vy, r, lfr)
    dalphafr_dvx = -(r*lfr + vy) / ( (vy + r*lfr)^2.0 +vx^2.0 )
end

function calc_dFyfr_dvx(Fzfr, lfr, alpha_fr, dalphafr_dvx, cornering_stiff, sample_Fz, l, rho, cla, vx)
    dFyfr_dvx = cornering_stiff*(-Fzfr * dalphafr_dvx +alpha_fr*lfr*rho*cla*vx/l)/ sample_Fz
end

function calc_dFyfr_dvy()
end

function bicycle_model_est_p_spikes!(dupcdp, upcdp, p_stat, t)
    # static params of the car
    m, l, lf, lr, sample_fz, rho, g = p_stat

    # unpack variables
    # states
    u = upcdp[1:7]
    # unpack variables
    x, y, psi, vx, vy, r, steer = u[1:7]

    # dynamic parameters of the car
    Iz = upcdp[8]
    cornering_stiff = upcdp[9]
    cla = upcdp[10]
    p = [m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g]

    # accel command, commanded steer rate
    command = upcdp[11:12]
    D, delta = command

    # estimate normal
    FzF, FzR = normal_force(u, command, p) # TODO

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

    # Calculate derivatives wrt p
    dFyf_dcla = calc_dFyfr_dcla(vx, cornering_stiff, alpha_f, l, lf, sample_fz, rho)
    dFyr_dcla = calc_dFyfr_dcla(vx, cornering_stiff, alpha_r, l, lr, sample_fz, rho)
    dFyf_dcs = calc_dFyfr_dcs(alpha_f, FzF, sample_fz)
    dFyr_dcs = calc_dFyfr_dcs(alpha_r, FzR, sample_fz)
    # dayaw/dIz
    dayaw_dIz = -a_yaw/Iz
    # dayaw/dcs
    dayaw_dcs = ( lf*cos(steer)*dFyf_dcs - lr*dFyr_dcs )/Iz
    # dayaw/dcla
    dayaw_dcla = ( lf*cos(steer)*dFyf_dcla - lr*dFyr_dcla )/Iz
    # day/dcs
    day_dcs = ( cos(steer)*dFyf_dcs + dFyr_dcs )/m
    # day/dcla
    day_dcla = ( cos(steer)*dFyf_dcla + dFyr_dcla )/m
    # dax/dcs
    dax_dcs = sin(steer) * dFyf_dcs / m
    # dax/dcla
    dax_dcla = sin(steer) * dFyf_dcla / m

    # Calculate derivatives wrt u

    dx_dpsi = -vx*sin(psi) - vy*cos(psi)
    dx_dvx = cos(psi)
    dx_dvy = -sin(psi)
    dy_dpsi = vx*cos(psi) - vy*sin(psi)
    dy_dvx = -dx_dvy
    dy_dvy = dx_dvx

    # calculate alpha derivatives
    dalphaf_dvy = calc_dalphafr_dvy(vx, vy, r, lf)
    dalphar_dvy = calc_dalphafr_dvy(vx, vy, -r, lr)
    # dalphaf_dr = lf * dalphaf_dvy
    # dalphar_dr = - lr * dalphar_dvy
    dalphaf_dvx = calc_dalphafr_dvx(vx, vy, r, lf)
    dalphar_dvx = calc_dalphafr_dvx(vx, vy, -r, lr)

    # Calculate dFzfr intermediate derivatives
    dFzf_dvx = -lf*rho*cla*vx/l
    dFzr_dvx = -lr*rho*cla*vx/l

    # calculate dFyfr intermediate derivatives
    dFyf_dalphaf = -cornering_stiff * FzF / sample_fz
    dFyr_dalphar = -cornering_stiff * FzR / sample_fz
    dFyf_dFzf = -cornering_stiff * alpha_f / sample_fz
    dFyr_dFzr = -cornering_stiff * alpha_r / sample_fz
    # calculate dFyfr derivatives
    dFyf_dvx = dFyf_dalphaf*dalphaf_dvx + dFyf_dFzf*dFzf_dvx
    dFyr_dvx = dFyr_dalphar*dalphar_dvx + dFyr_dFzr*dFzr_dvx
    dFyf_dvy = dFyf_dalphaf*dalphaf_dvy
    dFyr_dvy = dFyr_dalphar*dalphar_dvy
    # dFyf_dr = lf * dFyr_dvy
    # dFyr_dr = -lr * dFyr_dvy
    dFyf_dr = lf * dFyf_dalphaf*dalphaf_dvy
    dFyr_dr = -lr * dFyr_dalphar*dalphar_dvy

    # calculate ax derivatives
    dax_dsteer = ( -F_xf*sin(steer)+F_yf*cos(steer)+sin(steer)*dFyf_dalphaf )/m
    dax_dFyf = sin(steer)/m
    dax_dr = vy + dax_dFyf*dFyf_dr
    dax_dvx = dax_dFyf*dFyf_dvx
    dax_dvy = r + dax_dFyf*dFyf_dvy

    # calculate ay derivatives
    day_dsteer = (-F_xf*cos(steer)-F_yf*sin(steer)+cos(steer)*dFyf_dalphaf )/m
    day_dFyf = cos(steer)/m
    day_dr = -vx + day_dFyf*dFyf_dr + dFyr_dr/m
    day_dvx = -r + day_dFyf*dFyf_dvx + dFyr_dvx/m
    day_dvy = day_dFyf*dFyf_dvy + dFyr_dvy/m

    # calculate ayaw derivatives
    dayaw_dsteer = lf*( -F_xf*cos(steer)-F_yf*sin(steer)+cos(steer)*dFyf_dalphaf )/Iz
    dayaw_dFyf = lf*cos(steer)/Iz
    dayaw_dr = dayaw_dFyf*dFyf_dr - lr*dFyr_dr/Iz
    dayaw_dvx = dayaw_dFyf*dFyf_dvx - lr*dFyr_dvx/Iz
    dayaw_dvy = dayaw_dFyf*dFyf_dvy - lr*dFyr_dvy/Iz

    Jacobian = zeros((7,12))
    Jacobian[1,3] = dx_dpsi
    Jacobian[1,4] = dx_dvx
    Jacobian[1,5] = dx_dvy
    Jacobian[2,3] = dy_dpsi
    Jacobian[2,4] = dy_dvx
    Jacobian[2,5] = dy_dvy
    Jacobian[3,6] = 1.0
    Jacobian[4,4] = dax_dvx
    Jacobian[4,5] = dax_dvy
    Jacobian[4,6] = dax_dr
    Jacobian[4,7] = dax_dsteer
    Jacobian[5,4] = day_dvx
	Jacobian[5,5] = day_dvy
	Jacobian[5,6] = day_dr
	Jacobian[5,7] = day_dsteer
    Jacobian[6,4] = dayaw_dvx
	Jacobian[6,5] = dayaw_dvy
	Jacobian[6,6] = dayaw_dr
	Jacobian[6,7] = dayaw_dsteer

    Jacobian[4,9] = dax_dcs
    Jacobian[4,10] = dax_dcla
    Jacobian[5,9] = day_dcs
    Jacobian[5,10] = day_dcla
    Jacobian[6,8] = dayaw_dIz
    Jacobian[6,9] = dayaw_dcs
    Jacobian[6,10] = dayaw_dcla

    return Jacobian

    # # bicycle model
    # dupcdp[1] = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    # dupcdp[2] = vx * sin(psi) + vy * cos(psi)
    # dupcdp[3] = r
    # dupcdp[4] = ax
    # dupcdp[5] = ay
    # dupcdp[6] = a_yaw
    # dupcdp[7] = delta

end

# Check to make sure the derivatives are calculated correctly
#=
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

u_ex = [0.0, 0.0, pi/2, 5.1, 2.3, 0.01, -2.0]
dp_ex = [520.0, 24000.0, -0.2]
com_ex = [4.2, 0.03]
udpc_ex = vcat(u_ex, dp_ex, com_ex)
dudpc_ex = zeros(12)
using ForwardDiff
@btime BM_Jacobian = ForwardDiff.jacobian(bicycle_model_est_p_Jacobian, udpc_ex)
@btime check_Jacobian = bicycle_model_est_p_spikes!(dudpc_ex, udpc_ex, p_stat, 0.0)
isapprox(BM_Jacobian[:,1:10],check_Jacobian[:,1:10])
=#
