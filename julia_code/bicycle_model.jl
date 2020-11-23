cd(@__DIR__)
using Pkg
Pkg.activate(".")

function tire_force(alpha, Fz, p) 
    -alpha .* p[6] .* Fz/p[7]
end
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

    dx = Array{Float32}(undef, 8)
    # u = [v_theta, accel, steer_target]
    # x = [x, y, psi, vx, vy, r, theta, steer_angle]

    # params of the car
    # mass, length, length front, length rear, yaw moment of inertia
    m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g = p

    # estimate normal
    FzF, FzR = normal_force(x, u, p) # TODO

    # unpack variables
    # x, y, psi, vx, vy, r, steer, T = x
    # above is from the paper, but let's use the matlab code instead
    x, y, psi, vx, vy, r, theta, steer = x
    v_theta, D, delta = u  # velocity along path, accel command, commanded steer rate 

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
    dx[1] = vx * cos(psi) - vy * sin(psi)  # sin/cos(x), x should be radians.. not sure if our data is in deg/rad
    dx[2] = vx * sin(psi) + vy * cos(psi)
    dx[3] = r
    dx[4] = ax
    dx[5] = ay
    dx[6] = a_yaw
    dx[7] = v_theta
    dx[8] = delta

    return dx
end

function gen_data(x0, g, p, collect_at=0:1:10)
    len=length(collect_at)
    xs=Array{Float32,2}(undef, len,8)
    xs[1,:]=x0
    dxs=Array{Float32,2}(undef, len,8)
    inputs=Array{Float32,2}(undef, len,3)

    for itr in 2:len
        inputs[itr-1,:]=g(collect_at[itr-1])
        dxs[itr,:]=bicycle_model(xs[itr-1,:], inputs[itr-1,:], p)
        xs[itr,:].=xs[itr-1].+dxs[itr]
    end

    return [collect_at, xs, dxs, inputs]

end


function example_g(t)
    
    [sin(t), t, cos(t)]

end

x0 = [1,2,1,3,1,1,1,1]
p = [1292.2, 3,1.006,1.534,0.5,8,0.5,0.1,0.2,0.1]
data = gen_data(x0,example_g,p)

using Plots
plot(data[1],data[2][:,1])
plot!(data[1],data[2][:,2])