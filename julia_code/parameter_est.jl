cd(@__DIR__)
# Depends on generate_data.jl which depends on bicycle_model.jl
include("generate_data.jl")

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
    cornering_stiff, Iz, cla = p_command[1]
    upc[8] = cornering_stiff
    upc[9] = Iz
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

function update_cla(cla,t)

end

function update_Iz(Iz,t)

end

function update_cornering_stiffness(cornering_stiffness,t)

end

function update_p(f,dyn_p,t)

end

minmax_delta = 30.0*pi/360.0 # max change in delta is 15 degrees
command_lb = [-20.0, -minmax_delta]
command_ub = [15.0, minmax_delta]
command_s = SobolSeq(command_lb, command_ub)

# function update_commands(u,t)
#     vx = u[4]
#     vy = u[5]
#     v = sqrt(vx^2.0+vy^2.0)
#
# end

function update_commands(u,t)
    vx = u[4]
    vy = u[5]
    v = sqrt(vx^2.0+vy^2.0)

end

test_dupc = zeros(12)
test_upc = ones(12)
p_stat = [350.0,3.0,1.5,1.5,3430.0,1.2,9.8]
test_p_command = [[3.0,4,5],[6,7]]

bicycle_model_p!(test_dupc, test_upc, p_stat, test_p_command)

function diffeq_bicycle_model_p(dupc, upc, p_stat,t)
    bicycle_model!(du, u, p, com(t))
end

test_dupc
test_upc
