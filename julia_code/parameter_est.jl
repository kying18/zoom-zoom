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


#= Check to see if there's any type inference issues with update_Iz
test_Iz0 = 550.0
test_Iz_max = 600.0
test_Iz_rate = 0.01
test_t = 0.0
@code_warntype update_Iz(test_Iz0, test_Iz_max, test_Iz_rate, test_t)
=#

#= Plot a test trajectory of Iz
plot(test_tspan,
    update_Iz.(550.0, 600.0, 0.01, test_tspan),
    title = "Evolution of Iz Over 5 Periods",
    xlabel = "t",
    ylabel = "Iz",
    legend = false)
=#

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

#=
#Check to see if there's any type inference issues with update_Iz
test_cs0 = 50000.0
test_cs_lin = -0.0005
test_cs_sine = 0.05
test_period = 90.0
test_t = 0.0
@code_warntype update_cornering_stiff(test_cs0, test_cs_lin, test_cs_sine, test_period, test_t)
=#

#= Plot a test trajectory of cornering stiffness
test_tspan = range(0, 450.0, length = 901)
plot(test_tspan,
    update_cornering_stiff.(50000.0, -0.0005, 0.05, 90, test_tspan),
    title = "Evolution of Cornering Stiffness Over 5 Periods",
    xlabel = "t",
    ylabel = "Cornering Stiffness",
    legend = false)
=#

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

#=
#Check to see if there's any type inference issues with update_Iz
test_d_cla = -0.2
test_period = 90.0
test_t = 0.0
@code_warntype update_cla(test_d_cla, test_period, test_t)
=#

#= Plot a test trajectory of cla
test_tspan = range(0, 90.0, length = 181)
plot(test_tspan,
    update_cla.(-0.05, 90.0, test_tspan),
    title = "Evolution of Cla Over 1 Period",
    xlabel = "t",
    ylabel = "Cla")
=#

# A function to update all the parameters over time
function update_p(p_update, t)
    new_p = Vector{Float64}(undef,3)
    Iz0, Iz_max, Iz_rate, cs0, cs_lin, cs_sine, period, d_cla = p_update
    new_p[1] = update_Iz(Iz0, Iz_max, Iz_rate, t)
    new_p[2] = update_cornering_stiff(cs0, cs_lin, cs_sine, period, t)
    new_p[3] = update_cla(d_cla, period, t)
    return new_p
end

#=
# Check update_p
test_p_update = [test_Iz0, test_Iz_max, test_Iz_rate, test_cs0, test_cs_lin, test_cs_sine, test_period, test_d_cla]
test_t = 0.0
@code_warntype update_p(test_p_update, test_t)
=#

# Find the commands used to generate the data with real parameters
function find_command(data_sol, c_step, s_step, t)
    i = Int( round(t/c_step)*(c_step/s_step+1) + 1 )
    return data_sol[11:12,i]
end

#=
# Check find_command
@code_warntype find_command(spike_sol, c_step, s_step, 0.0)
=#

#=
Parameter estimation df/dp calculations
=#
function calc_dFyfr_dcla(vx, cornering_stiff, alphafr, l, lfr, sample_Fz, rho)
    dFyfr_dcla = 0.5 * alphafr * cornering_stiff * lfr * rho * vx^2.0 / (sample_Fz * l)
    return dFyfr_dcla
end

#=
# Check calc_dFyfr_dcla
@code_warntype calc_dFyfr_dcla(5.0, cs0, 0.0, l, lf, sample_fz, rho)
=#

function calc_dFyfr_dcs(alphafr, Fzfr, sample_Fz)
    dFyfr_dcs = -alphafr * Fzfr / sample_Fz
    return dFyfr_dcs
end

#=
# Check calc_dFyfr_dcs
@code_warntype calc_dFyfr_dcs(0.0, 10.0, sample_fz)
=#


#=
Parameter estimation df/du calculations
=#
function calc_dalphafr_dvy(vx, vy, r, lfr)
    dalphafr_dvy = vx / ( (vy + r*lfr)^2.0 +vx^2.0 )
end

#=
# Check calc_dalphafr_dvy
@code_warntype calc_dalphafr_dvy(4.0, 5.0, 1.0, lf)
=#

function calc_dalphafr_dvx(vx, vy, r, lfr)
    dalphafr_dvx = -(r*lfr + vy) / ( (vy + r*lfr)^2.0 +vx^2.0 )
end

#=
# Check calc_dalphafr_dvx
@code_warntype calc_dalphafr_dvx(4.0, 5.0, 1.0, lf)
=#

# Check to make sure the derivatives are calculated correctly
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

function bicycle_model_est_p!(dupcdp, upcdp, p_stat, t)
    # static params of the car
    m, l, lf, lr, sample_fz, rho, g = p_stat

    # Initialize an array for the derivatives
    dBM_du = zeros((6,6))
    dBM_du[3,6] = 1.0
    dBM_dp = zeros((6,3))

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

    # Update derivatives of bicycle model wrt u
    dBM_du[1,3] = dx_dpsi
    dBM_du[1,4] = dx_dvx
    dBM_du[1,5] = dx_dvy
    dBM_du[2,3] = dy_dpsi
    dBM_du[2,4] = dy_dvx
    dBM_du[2,5] = dy_dvy
    # dBM_du[3,6] = 1.0
    dBM_du[4,4] = dax_dvx
    dBM_du[4,5] = dax_dvy
    dBM_du[4,6] = dax_dr
    # dBM_du[4,7] = dax_dsteer
    dBM_du[5,4] = day_dvx
    dBM_du[5,5] = day_dvy
    dBM_du[5,6] = day_dr
    # dBM_du[5,7] = day_dsteer
    dBM_du[6,4] = dayaw_dvx
    dBM_du[6,5] = dayaw_dvy
    dBM_du[6,6] = dayaw_dr
    # dBM_du[6,7] = dayaw_dsteer

    # Update derivatives of bicycle model wrt u
    dBM_dp[4,2] = dax_dcs
    dBM_dp[4,3] = dax_dcla
    dBM_dp[5,2] = day_dcs
    dBM_dp[5,3] = day_dcla
    dBM_dp[6,1] = dayaw_dIz
    dBM_dp[6,2] = dayaw_dcs
    dBM_dp[6,3] = dayaw_dcla

    # # bicycle model
    dupcdp[1] = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    dupcdp[2] = vx * sin(psi) + vy * cos(psi)
    dupcdp[3] = r
    dupcdp[4] = ax
    dupcdp[5] = ay
    dupcdp[6] = a_yaw
    dupcdp[7] = delta
    # dupcdp[13:33] .= vec(dBM_du * reshape(@view(upcdp[13:33]),(7,3)) + dBM_dp)
    dupcdp[13:30] .= vec(dBM_du * reshape(@view(upcdp[13:30]),(6,3)) + dBM_dp)

end


#=
# Check bicycle_model_est_p!
test_upc = [0.0,0.0,5.0,0.0,0.0,0.0,0.0, Iz0, cs0, cla0, D0, delta0]
test_dp = zeros(18)
test_upcdp = vcat(test_upc, test_dp)
test_dupcdp = zeros(30)
test_t = 0.0
test_pstat = [m, l, lf, lr, sample_fz, rho, g]
@code_warntype bicycle_model_est_p!(test_dupcdp, test_upcdp, test_pstat, test_t)
=#


# A function to calculate the gradient of the cost
function calc_dCdp(p_est_sol, data_sol)
    # u_diff = p_est_sol[1:7,:]-data_sol[1:7,1:length(p_est_sol)]
    u_diff = p_est_sol[1:6,:]-data_sol[1:6,1:length(p_est_sol)]
    C = 0.0
    dCdp = zeros(3)
    for i in 1:length(p_est_sol)
        # Calculate the cost
        C = C + sum(u_diff[:,i].^2)
        # Calculate the gradient of the cost
        dCdp = dCdp + transpose(reshape(p_est_sol[13:30,i],(6,3)))*u_diff[:,i]
    end
    dCdp = dCdp*2.0
    return C, dCdp
end

#=
# Check bicycle_model_est_p!
@code_warntype calc_dCdp(stat_sol, spike_sol)
=#

"""
A function to calculate the cost of the bicycle model vs the "real data" for use
with estimating dCdp with ForwardDiff.
"""
function cost_Jacobian(dyn_p)
    t0 = 0.0
    tf = 2.5
    upc0 = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, dyn_p[1], dyn_p[2], dyn_p[3], 0.5, 0.01]
    dp_0 = zeros(18)
    upcdp0 = vcat(upc0, dp_0)
    p_est_tspan = (t0,tf)
    p_est_dosetimes = t0:c_step:tf
    p_est_affect!(integrator) = integrator.u[11:12] .= find_command(spike_sol, c_step, s_step, integrator.t)
    p_est_cb = PresetTimeCallback(p_est_dosetimes, p_est_affect!)
    p_est_prob=ODEProblem(bicycle_model_est_Jacobian!, upcdp0, p_est_tspan, p_stat)
    p_est_sol = solve(p_est_prob,callback=p_est_cb, Tsit5(), dt = 0.01, adaptive=false, saveat=s_step)
    u_diff = p_est_sol[1:6,:]-spike_sol[1:6,1:length(p_est_sol)]
    C = 0.0
    for i in 1:length(p_est_sol)
        # Calculate the cost
        C = C + sum(u_diff[:,i].^2)
    end
    return C
end

"""
A function to calculate the bicycle model for use with estimating dCdp with
ForwardDiff
"""
function bicycle_model_est_Jacobian!(dupcdp, upcdp, p_stat, t)
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

    # # bicycle model
    dupcdp[1] = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    dupcdp[2] = vx * sin(psi) + vy * cos(psi)
    dupcdp[3] = r
    dupcdp[4] = ax
    dupcdp[5] = ay
    dupcdp[6] = a_yaw
    dupcdp[7] = delta

end


function gradient_descent_store(alpha, n_itr, data_sol, bounds, t_arr, upcdp0, p_stat)
    # Create an array to store the parameters
    p_array = Vector{Array{Float64,1}}(undef,n_itr+1)
    # Create an array to store the costs
    C_array = Vector{Float64}(undef,n_itr)
    # Create an array to work with
    upcdp = copy(upcdp0)
    # Unpack the time array
    t0, tf, com_step, save_step = time_arr

    # Unpack the bounds
    p_lb = bounds[1]
    u_lb = bounds[2]

    # Store the initial guess for p
    p_array[1] = upcdp0[8:10]
    p_est_tspan = (t0,tf)
    p_est_dosetimes = t0:com_step:tf
    p_est_affect!(integrator) = integrator.u[11:12] .= find_command(data_sol, com_step, save_step, integrator.t)
    p_est_cb = PresetTimeCallback(p_est_dosetimes, p_est_affect!)

    for i in 1:n_itr
        # Solve the ODE
        p_est_prob=ODEProblem(bicycle_model_est_p!, upcdp, p_est_tspan, p_stat)
        p_est_sol = solve(p_est_prob,callback=p_est_cb, Tsit5(), dt = 0.01, adaptive=false, saveat=save_step)
        # Calculate the cost and the gradient for each new p
        C, dCdp = calc_dCdp(p_est_sol, spike_sol)
        # Store the calculated cost
        C_array[i] = C
        # Compute the next p value
        p_next = p_array[i] - alpha .* vec(dCdp)
        # Make sure it fits in the bounds
        for j in 1:length(p_next)
            if p_next[j] < p_lb[j]
                p_next[j] = p_lb[j]
            elseif p_next[j] > p_ub[j]
                p_next[j] = p_ub[j]
            end
        end
        p_array[i+1]=p_next
        # Update the parameter values in the state vector
        upcdp[8:10] = p_array[i+1]

        # if i == n_itr
        #     plot(spike_sol, vars=1, title="X Trajectory: $index", xlabel="x", ylabel="t")
        #     plot!(p_est_sol, vars=1)
        #     savefig("X Trajectory: $index.png")
        #
        #     plot(spike_sol, vars=2, title="Y Trajectory: $index", xlabel="y", ylabel="t")
        #     plot!(p_est_sol, vars=2)
        #     savefig("y Trajectory: $index.png")
        # end

    end
    return([C_array, p_array])
end

function gradient_descent(alpha, n_itr, bounds, data_sol, time_arr, upcdp0, p_stat)
    # Create an array to store the parameters
    p_prev = upcdp0[8:10]
    # Create an array to store the costs
    C = 0.0
    # Create an array to work with
    upcdp = copy(upcdp0)

    # Unpack the bounds
    p_lb = bounds[1]
    u_lb = bounds[2]

    # Unpack the time array
    t0, tf, com_step, save_step = time_arr
    p_est_tspan = (t0,tf)
    p_est_dosetimes = t0:com_step:tf
    p_est_affect!(integrator) = integrator.u[11:12] .= find_command(data_sol, com_step, save_step, integrator.t)
    p_est_cb = PresetTimeCallback(p_est_dosetimes, p_est_affect!)

    for i in 1:n_itr
        # Solve the ODE
        p_est_prob = ODEProblem(bicycle_model_est_p!, upcdp, p_est_tspan, p_stat)
        # p_est_sol = solve(p_est_prob,callback=p_est_cb, Tsit5(), dt = 0.01, adaptive=false, saveat=s_step)
        p_est_sol = solve(p_est_prob,callback=p_est_cb, Tsit5(), dt = 0.01, adaptive=false, saveat=save_step)
        # Calculate the cost and the gradient for each new p
        C, dCdp = calc_dCdp(p_est_sol, spike_sol)

        # Compute the next p value
        p_next = p_prev - alpha .* vec(dCdp)
        # Make sure it fits in the bounds
        for j in 1:length(p_next)
            if p_next[j] < p_lb[j]
                p_next[j] = p_lb[j]
            elseif p_next[j] > p_ub[j]
                p_next[j] = p_ub[j]
            end
        end

        # Update the parameter values in the state vector
        upcdp[8:10] = p_next

        # Switch the parameter arrays
        p_prev, p_next = p_next, p_prev

    end
    return(C, p_prev)
end


#= Gradient Descent
# Generate parameter estimation data
upc0 = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, Iz0, cs0, cla0, D0, delta0]
dp_0 = zeros(18)
upcdp0 = vcat(upc0, dp_0)

p_lb = [540.0, 19000.0, -0.55]
p_ub = [610.0, 52000.0, -0.5]
est_bounds = [p_lb, p_ub]

p_est_tf = 1.0
time_array = [0.0, p_est_tf, c_step, s_step]
alpha_Iz = 0.005
alpha_cs = 0.005
alpha_cla = 0.0001
gd_rate = [alpha_Iz, alpha_cs, alpha_cla]
n_itr = 450
# test_C, test_p = gradient_descent_store(gd_rate, n_itr, est_bounds, spike_sol, time_array, upcdp0, p_stat, 0)
test_C, test_p = gradient_descent(gd_rate, n_itr, est_bounds, spike_sol, time_array, upcdp0, p_stat)


plot(test_C,
   title="Cost Gradient Descent Progression",
   xlabel="steps", ylabel="Cost",
   yaxis=:log,
   legend=false)

Iz_vals = [test_p[i][1] for i in 1:length(test_p)]
cs_vals = [test_p[i][2] for i in 1:length(test_p)]
cla_vals = [test_p[i][3] for i in 1:length(test_p)]
plot(Iz_vals, title="Iz Evolution", xlabel="steps", ylabel="Iz", legend=false)
plot(cs_vals, title="cs Evolution", xlabel="steps", ylabel="cs", legend=false)
plot(cla_vals, title="cla Evolution", xlabel="steps", ylabel="cla", legend=false)
=#

#=
Gradient Descent Profiling Analysis
# benchmark
@btime test_C, test_p = gradient_descent(gd_rate, n_itr, est_bounds, spike_sol, time_array, upcdp0, p_stat)
# Check types
@code_warntype gradient_descent(gd_rate, n_itr, est_bounds, spike_sol, time_array, upcdp0, p_stat)
=#

# Find the commands used to generate the data with real parameters
function find_upc(data_sol, c_step, s_step, t)
    i = Int( round(t/c_step)*(c_step/s_step+1) + 1 )
    return data_sol[1:12,i]
end

#=
# Check find_command for type stability
@code_warntype find_upc(spike_sol, c_step, s_step, 0.0)
=#

# A function to estimate the parameter evolution over time
function est_param(alpha, n_itr, bounds, time_arr, data_sol, dynp_est0, p_stat)
    # Unpack the time array
    dt, tspan, t0, tf, com_step, save_step = time_arr
    gd_time_arr = [t0, tf, com_step, save_step]
    # Create a time range
    t_range = t0:dt:tf-tspan

    # Unpack the bounds
    p_lb = bounds[1]
    p_ub = bounds[2]

    # Create an array to store the estimated parameters in
    est_p_array = Vector{Array{Float64,1}}(undef, length(t_range)-1)
    dyn_p_est = dynp_est0

    # Initialize a vector for upcdp
    upcdp_0 = zeros(30)

    for i in 1:length(t_range)-1
        # Figure out the initial and end time
        p_est_t0 = t_range[i]
        println(p_est_t0)
        p_est_tf = p_est_t0+tspan
        p_est_tspan = (p_est_t0, p_est_tf)

        # Figure out what the initial state of the vehicle is
        upcdp_0[1:12] = find_upc(spike_sol, com_step, save_step, p_est_t0)
        # Change the parameters to the estimated value
        upcdp_0[8:10] = dyn_p_est
        # Start du/dp at 0
        upcdp_0[13:30] = zeros(18)

        # Perform gradient descent
        cost, p_estimated = gradient_descent(alpha, n_itr, bounds, data_sol, gd_time_arr, upcdp_0, p_stat)
        # Update the estimated parameters
        dyn_p_est = p_estimated
        # Save the estimated parameters
        est_p_array[i] = dyn_p_est
    end
    return est_p_array
end

# Set up Sobol sequence for generating commands
minmax_delta = 30.0*pi/360.0 # max change in delta is 15 degrees
# minmax_delta = 15.0*pi/360.0 # max change in delta is 7.5 degrees
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
cs0 = 50000.0
cs_lin = -0.0005
cs_sine = 0.05
# period = 90.0
period = 360.0
d_cla = -0.05
cla0 = -0.5
# Assemble p_update
p_update = [Iz0, Iz_max, Iz_rate, cs0, cs_lin, cs_sine, period, d_cla]
# Assemble p_diffeq for the ODEproblem
p_diffeq = [p_stat, p_update]
# Define the initial upc vector
# [x, y, phi, vx, vy, r, steer, Iz, cornering_stiff, cla, D, delta]
D0 = 0.5
delta0 = 0.01
upc0 = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, Iz0, cs0, cla0, D0, delta0]
spike_tmax = 100.0
spike_tspan = (0.0,spike_tmax)
global c_step = 0.5
global s_step = 0.05
dosetimes = 0.0:c_step:spike_tmax
affect!(integrator) = integrator.u[11:12] .= next!(command_s)
cb = PresetTimeCallback(dosetimes,affect!)
spike_prob=ODEProblem(bicycle_model_p_spikes!, upc0, spike_tspan, p_diffeq)
spike_sol = solve(spike_prob,callback=cb, Tsit5(), dt = 0.01, adaptive=false, saveat=s_step)
#plot path of car
plot(spike_sol, vars=(1,2), xlabel="x", ylabel="y", title="True Vehicle Trajectory", label="true") # (x, y)
plot(spike_sol, vars=(1,2), xlabel="x", ylabel="y",
    title="True Vehicle Trajectory", legend = false,
    xlims=(-200,430),ylims=(-150,250)) # (x, y)
# savefig("Pest - True Trajectory.png")
plot(spike_sol, vars=8, xlabel="t", ylabel="Iz", title="Evolution of Iz", legend=false) # Iz
plot(spike_sol, vars=9, xlabel="t", ylabel="Cornering Stiffness", title="Evolution of Cornering Stiffness", legend=false) # Cornering stiffness
plot(spike_sol, vars=10, xlabel="t", ylabel="Cla", title="Evolution of Cla", legend=false)
plot(spike_sol, vars=11, xlabel="t", ylabel="D", title="Parameter Estimation Input Acceleration", legend=false)
plot(spike_sol, vars=12, xlabel="t", ylabel="delta", title="Parameter Estimation Input Steering", legend=false)

# Generate data with static parameters
u0_cb=[0.0,0.0,0.0,5.0,0.0,0.0,0.0,D0,delta0]
p_cb = [m, l, lf, lr, Iz0, cs0, sample_fz, rho, cla0, g]
stat_dosetimes = 0.0:c_step:spike_tmax
stat_affect!(integrator) = integrator.u[8:9] .= find_command(spike_sol, c_step, s_step, integrator.t)
stat_cb = PresetTimeCallback(dosetimes,stat_affect!)
stat_prob=ODEProblem(bicycle_model_callback!, u0_cb, (0.0,spike_tmax), p_cb)
stat_sol = solve(stat_prob, callback=stat_cb, Tsit5(), dt = 0.01, adaptive=false, saveat=s_step)

plot!(stat_sol, vars=(1,2),label="static", title="True vs Static XY Trajectory",
    legend= :bottomright, xlims=(-200,430),ylims=(-150,250)) # (x,y)
# savefig("Pest Slower - True vs Stat Trajectory.png")

#=
# Check to see if the gradients of the bicycle model wrt u and p are correct
u_ex = [0.0, 0.0, pi/2, 5.1, 2.3, 0.01, -2.0]
dyn_p_ex = [520.0, 24000.0, -0.2]
com_ex = [4.2, 0.03]
dp_0 = ones(21)
upc_ex = vcat(u_ex, dyn_p_ex, com_ex)
upcdp_ex = vcat(u_ex, dyn_p_ex, com_ex, dp_0)
dupcdp_ex = zeros(33)
dBM_du0 = zeros((6,6))
dBM_du0[3,6] = 1.0
dBM_dp0 = zeros((6,3))
p_est = [p_stat, dBM_du0, dBM_dp0]
using ForwardDiff
BM_Jacobian = ForwardDiff.jacobian(bicycle_model_est_p_Jacobian, upc_ex)
@btime check_Jacobian = bicycle_model_est_p!(dupcdp_ex, upcdp_ex, p_est, 0.0)
check_Jacobian = bicycle_model_est_p!(dupcdp_ex, upcdp_ex, p_est, 0.0)
isapprox(BM_Jacobian[1:6,1:6],dBM_du0)
isapprox(BM_Jacobian[1:6,8:10],dBM_dp0)
=#

#= Check cost gradient
# Check to make sure dC/dp is computed correctly
cost_Jacobian([Iz0, cs0, cla0])
using ForwardDiff
ForwardDiff.gradient(cost_Jacobian,[Iz0, cs0, cla0])

upc0 = [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, Iz0, cs0, cla0, D0, delta0]
dp_0 = zeros(18)
upcdp0 = vcat(upc0, dp_0)
dBM_du0 = zeros((6,6))
dBM_du0[3,6] = 1.0
dBM_dp0 = zeros((6,3))
p_est = [p_stat, dBM_du0, dBM_dp0]
p_est_t0 = 0.0
p_est_tf = 2.5
p_est_tspan = (p_est_t0,p_est_tf)
p_est_dosetimes = p_est_t0:c_step:p_est_tf
p_est_affect!(integrator) = integrator.u[11:12] .= find_command(spike_sol, c_step, s_step, integrator.t)
p_est_cb = PresetTimeCallback(p_est_dosetimes, p_est_affect!)
p_est_prob=ODEProblem(bicycle_model_est_p!, upcdp0, p_est_tspan, p_est)
p_est_sol = solve(p_est_prob,callback=p_est_cb, Tsit5(), dt = 0.01, adaptive=false, saveat=s_step)
calc_dCdp(p_est_sol, spike_sol)
=#

# Estimate parameters
dynp_est0 = [Iz0, cs0, cla0]
p_lb = [540.0, 19000.0, -0.55]
p_ub = [610.0, 52000.0, -0.5]
est_bounds = [p_lb, p_ub]
dt = 1.0
est_tspan = 1.0
est_t0 = 0.0
est_tf = 5.0
est_time_arr = [dt, est_tspan, est_t0, est_tf, c_step, s_step]
t_plot = est_tspan/2:dt:est_tf-dt-est_tspan
alpha_Iz = 0.005
alpha_cs = 0.5
alpha_cla = 0.0005
gd_rate = [alpha_Iz, alpha_cs, alpha_cla]
n_itr = 450
estimated_params = est_param(gd_rate, n_itr, est_bounds, est_time_arr, spike_sol, dynp_est0, p_stat)

#=
# Parameter Estimation Benchmarking and Profiling
# Check type stability
@code_warntype est_param(gd_rate, n_itr, est_bounds, est_time_arr, spike_sol, dynp_est0, p_stat)
# Check profile
@profile est_param(gd_rate, n_itr, est_bounds, est_time_arr, spike_sol, dynp_est0, p_stat)
Juno.profiler()
=#

# Pick out estimated parameters
estimated_Iz_vals = [estimated_params[i][1] for i in 1:length(estimated_params)]
estimated_cs_vals = [estimated_params[i][2] for i in 1:length(estimated_params)]
estimated_cla_vals = [estimated_params[i][3] for i in 1:length(estimated_params)]

# Plot the true vs estimated parameters over time
plot(spike_sol, vars=8, xlabel="t", ylabel="Iz", title="True vs Estimated Iz",
    label="true", legend= :topleft) # Iz
plot!(t_plot, estimated_Iz_vals, label="est", linecolor="red")
# savefig("Pest Slower - True vs Estimated Iz.png")

plot(spike_sol, vars=9, xlabel="t", ylabel="Cornering Stiffness",
    title="True vs Estimated Cornering Stiffness", label = "true",
    legend = :topleft) # Cornering stiffness
plot!(t_plot, estimated_cs_vals, label="est", linecolor="red")
# savefig("Pest Slower - True vs Estimated cs.png")

plot(spike_sol, vars=10, xlabel="t", ylabel="Cla", title="True vs Estimated Cla",
    label="true", legend= :outerright)
plot!(t_plot, estimated_cla_vals, label="est", linecolor="red")
# savefig("Pest Slower - True vs Estimated cla.png")

# Plot the xy trajectory of the true vs estimated solution
plot(spike_sol, vars=(1,2), xlabel="x", ylabel="y", label = "true",
    legend = :bottomright)
plot!(xlims=(-200,430),ylims=(-150,300), title="True Trajectory", legend=false)
# savefig("Pest Slower - True Trajectory.png")

# Compute and plot the xy trajectory of the estimated solution
plot_est_dosetimes = est_t0:c_step:est_tf
plot_est_affect!(integrator) = integrator.u[11:12] .= find_command(spike_sol, c_step, s_step, integrator.t)
plot_est_cb = PresetTimeCallback(plot_est_dosetimes, plot_est_affect!)
t_range = est_t0:dt:est_tf-est_tspan
upcdp_est = zeros(30)
plot_p_est = [p_stat, dBM_du0, dBM_dp0]
# plot([], xlabel="x", ylabel="y", title="Estimated XY Trajectory", legend=false)

# Calculate the L2 loss
L2_est = 0.0
for i in 1:length(t_range)-1
    # Figure out the initial and end time
    plot_est_t0 = t_range[i]
    plot_est_tf = plot_est_t0 + est_tspan

    plot_est_tspan = (plot_est_t0, plot_est_tf)
    # Figure out what upcdp
    upcdp_est[1:12] = find_upc(spike_sol, c_step, s_step, plot_est_t0)
    # Change the parameters to the estimated value
    upcdp_est[8:10] = estimated_params[i]
    plot_est_prob=ODEProblem(bicycle_model_est_p!, upcdp_est, plot_est_tspan, plot_p_est)
    plot_est_sol = solve(plot_est_prob, callback=plot_est_cb, Tsit5(), dt = 0.01, adaptive=false, saveat=s_step)

    # Calculate the total L2 loss
    j = Int( round(plot_est_t0/c_step)*(c_step/s_step+1) + 1 )
    diff = plot_est_sol[1:2,:]-spike_sol[1:2,j:j+length(plot_est_sol)-1]
    for d in 1:length(plot_est_sol)
        global L2_est = L2_est + sum(diff[d].^2)
    end

    # Plot
    if i != length(t_range)-1
        plot!(plot_est_sol, vars=(1,2), linecolor="red", label=false)
    else
        plot!(plot_est_sol, vars=(1,2), linecolor="red", label="est")
    end
end

plot!(title="True vs Estimated XY Trajectory", legend=:bottomright,
    xlims=(-200,430),ylims=(-150,250))
# savefig("Pest Slower - True vs Estimated XY Trajectory.png")

# Look at the L2 loss of the xy position for the estimated solution
L2_est
println(L2_est)

# Look at the L2 loss of the static parameter solution
L2_stat = 0.0
stat_diff = stat_sol[1:2,:]-spike_sol[1:2,:]
for d in 1:length(stat_sol)
    global L2_stat = L2_stat + sum(stat_diff[d].^2)
end
println(L2_stat)
