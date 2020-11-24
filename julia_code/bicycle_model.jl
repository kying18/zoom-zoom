cd(@__DIR__)
using Pkg
Pkg.activate(".")
using Random
using Distributions

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

"""
    percent_noise_flat(x, percent_noise)

Computes noise by adding or subtracting a specified percentage of the provided
x array as error: noisy_x = x +/- (percent)*x
"""
function percent_noise_flat(x, percent_noise)
    # Create array of +/-1 the size of x
    noise = rand([Float32(-1.0),Float32(1.0)],length(x))
    # Multiply the percent noise by the x values, and then multiply by the
    # +/- noise array. Add to the x values.
    noisy_x = x + percent_noise * x .* noise
    return noisy_x
end

"""
    percent_noise_normal(x, percent_noise)

Adds normally distributed noise by calculating noise N(0.0, percent_noise) and
computing noisy_x = x + N(0.0, percent_noise)*x.
"""
function percent_noise_normal(x, percent_noise)
    # Create an array the size of x
    noisy_x = Vector{Float32}(undef,length(x))
    # For each x, sample from the N(x, percent*x) distribution
    for i in 1:length(x)
        noisy_x[i] = x[i] * (Float32(1.0) + rand(Normal(0.0,percent_noise)))
    end
    return noisy_x
end

test_xs[1,:]
percent_noise_normal(test_xs[1,:],0.05)
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
    new_params = Vector{Float32}(undef,10)
    # Calculate the previous change in parameters to figure out its sign later
    if i > 2
        prev_dp = params[i-1,:] - params[i-2,:]
    else
        # Set prev_dp to an array of -1.0s for the first parameter change
        prev_dp = -1.0 * ones(Float32,length(params[i-1,:]))
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

function gen_data(x0, g, p, collect_at=0:1:10; gen_params=false,
    percent_dp=1e-4*ones(Float32,10), add_noise=false, percent_noise=0.01)
    # Figure out number of data points to collect
    len=length(collect_at)
    # Array to store the state data in
    xs=Array{Float32,2}(undef,len,8)
    # Save the initial state
    xs[1,:]=x0
    # Arrays to store dxs and inputs in
    dxs=Array{Float32,2}(undef, len,8)
    inputs=Array{Float32,2}(undef, len,3)

    # Create storage array if parameters are being modified
    if gen_params != false
        # Create an array to store parameters in
        params=Array{Float32,2}(undef,len, length(p))
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

function example_g(t)

    [sin(t), t, cos(t)]

end

x0 = [1,2,1,3,1,1,1,1]
p = [1292.2, 3,1.006,1.534,0.5,8,0.5,0.1,0.2,0.1]
data = gen_data(x0,example_g,p)
normal_noisy_data = gen_data(x0,example_g,p,add_noise=percent_noise_normal)
normal_noisy_data = gen_data(x0,example_g,p,add_noise=percent_noise_flat)
normal_noisy_data = gen_data(x0,example_g,p,gen_params=change_params_percent_normal)
normal_noisy_data = gen_data(x0,example_g,p,add_noise=percent_noise_normal,gen_params=change_params_percent_normal)

using Plots
plot(data[1],data[2][:,1])
plot!(data[1],data[2][:,2])
