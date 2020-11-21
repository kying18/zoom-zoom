cd(@__DIR__)
using Pkg
Pkg.activate(".")

tire_force(alpha, Fz, p) = -alpha .* p[6] .* Fz/p[7]

function normal_force(x, u, p)
    m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g = p
    # Traction due to aero
    vx = x[4]; # Lon Velocity
    F_lift = 0.5 * rho * cla * vx^2;
    FzF = -( lr * (m * g + F_lift))/ l;
    FzR = -( lf*(m * g + F_lift)) / l;

    return FzF, FzR
end

# x = [x, y, psi, vx, vy, r, delta, T]
function bicycle_model(x, u, p)
    # u = [v_theta, accel, steer_target]
    # x = [x, y, psi, vx, vy, r, theta, steer_angle]

    # params of the car
    # mass, length, length front, length rear, yaw moment of inertia
    m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g = p

    # unpack variables
    # x, y, psi, vx, vy, r, steer, T = x
    # above is from the paper, but let's use the matlab code instead
    x, y, psi, vx, vy, r, theta, steer = x
    v_theta, D, delta = u  # velocity along path, accel command, commanded steer rate 

    # compute slip angles
    alpha_f = atan((vy + r * lf) / vx) + steer
    alpha_r = atan((vy - r * lr) / vx)

    # estimate normal
    FzF, FzR = normal_force(x, u, p) # TODO

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
    dx[1] = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    dx[2] = vx * sin(psi) + vy * cos(psi)
    dx[3] = r
    dx[4] = ax
    dx[5] = ay
    dx[6] = a_yaw
    dx[7] = v_theta
    dx[8] = delta
end