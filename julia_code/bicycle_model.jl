cd(@__DIR__)
using Pkg
Pkg.activate(".")

function tire_force(alpha, Fz, p)
    -alpha .* p[6] .* Fz/p[7]
    # p[6] is cornering stiffness
    # p[7] is fz
end

function normal_force(u, command, p)
    m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g = p
    # Traction due to aero
    vx = u[4]; # Lon Velocity
    F_lift = 0.5 * rho * cla * vx^2;
    FzF = -( lr * (m * g + F_lift))/ l;
    FzR = -( lf*(m * g + F_lift)) / l;
    return FzF, FzR
end

"""
Mutating bicycle model to use with DifferentialEquations.jl Takes in u, a vector
of the states and the commands. (9x1)
"""
function bicycle_model!(du, u, p, command)

    # params of the car
    m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g = p
    # lf/lr are distances between center of mass and front/rear axel
    # Iz is yaw moment of inertia
    #


    # estimate normal
    FzF, FzR = normal_force(u, command, p) # TODO

    # unpack variables
    x, y, psi, vx, vy, r, steer = u[1:7]
    D, delta = command  # velocity along path, accel command, commanded steer rate
    u[8]=D
    u[9]=delta
    # compute slip angles
    # alpha_f = front slip angle ---- alpha_r = rear slip angle
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
    du[1] = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    du[2] = vx * sin(psi) + vy * cos(psi)
    du[3] = r
    du[4] = ax
    du[5] = ay
    du[6] = a_yaw
    du[7] = delta

end

"""
Non-mutating bicycle model to use with random data generation. Takes in u, a
vector of the states. (7x1)
"""
function bicycle_model(u, p, command)
    # Create an array to store du in.
    du = Vector{Float64}(undef, length(u))

    # params of the car
    m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g = p

    # estimate normal
    FzF, FzR = normal_force(u, command, p) # TODO

    # unpack variables
    x, y, psi, vx, vy, r, steer = u[1:7]
    D, delta = command  # velocity along path, accel command, commanded steer rate

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
    du[1] = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    du[2] = vx * sin(psi) + vy * cos(psi)
    du[3] = r
    du[4] = ax
    du[5] = ay
    du[6] = a_yaw
    du[7] = delta
    return du
end

#=
u= [x = global x value ~ 0
    y = global y value ~ 0
    psi = global car aligment yaw value ~ 0
    vx = forward velocity of the car along its axis in global refrence frame ~ 5
    vy = sideways velocity of the car along its axis in global refrence frame ~ 0
    r = global yaw rate (dpsi/dt) ~0
    steer = car steering angle relative to its axis ~ 0 between -.26 and 0.26
    com_store1 = command accel at time step
    com_store2 = command steer_rate at time step
    ]
=#
u_ex=[0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0]

#=
p= [
    m = mass ~ 350kg
    l = car length ~ 3m
    lf = forward dist from center of mass ~ 1.5m
    lr = rear dist from  center of mass ~ 1.5m
    Iz = yaw moment of inertia ~ 550
    cornering_stiff = sample cornering stiffness ~ 20,00 or 10,000
    sample_fz = sample downward force at observed cornering stiffness ~ 3430
    rho = density of air ~ 1.2
    cla = downforce for velocity ~ +/- 0.5 (should increase downforce with speed)
    g = gravity ~ 9.8 m/s/s
    ]
=#
p_ex=[350.0,3.0,1.5,1.5,550.0,10000.0,3430.0,1.2,-0.5,9.8]

#=
command(t) = [
    accel = global acceleration ~ from -20 to 15 m/s/s
    steer_rate = steer rate along car axis from -0.26 to 0.26
    ]
=#

# added sin curve for steering rate
#[accel, steer_rate]
function com(t)
    [5,0.1*sin(t)] .* 2 .*(rand(Float64, (2)).-0.5)
end

function diffeq_bicycle_model(du, u, p,t)
    bicycle_model!(du, u, p, com(t))
end
