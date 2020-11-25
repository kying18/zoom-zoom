cd(@__DIR__)
using Pkg
Pkg.activate(".")
using Random
using Distributions
using DifferentialEquations

function tire_force(alpha, Fz, p) 
    -alpha .* p[6] .* Fz/p[7]
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


#max tire force = mass*10-15 ms/s^2




function bicycle_model!(du, u, p, command)


    # params of the car
    m, l, lf, lr, Iz, cornering_stiff, sample_fz, rho, cla, g = p

  
    # estimate normal
    FzF, FzR = normal_force(u, command, p) # TODO

    # unpack variables
    x, y, psi, vx, vy, r, steer = u[1:7]
    D, delta = command  # velocity along path, accel command, commanded steer rate 
    u[8]=D
    u[9]=delta
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

end

#=
command(t) = [
    accel = global acceleration ~ from -20 to 15 m/s/s
    steer_rate = steer rate along car axis ~ ?
    ]
=#
function com(t)
    [5,0.1]
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
u0=[0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0]

#=
p= [
    m = mass ~ 350kg
    l = car length ~ 3m
    lf = forward dist from center of mass ~ 1.5m
    lr = rear dist from  center of mass ~ 1.5m
    Iz = yaw moment of inertia ~ 550
    cornering_stiff = sample cornering stiffness ~ 20,00 or 10,000
    sample_fz = sample downard force at observed cornering stiffness ~ 3500
    rho = density of air ~ 1.2
    cla = downforce for velocity ~ +/- 0.5 (should increase downforce with sped)
    g = gravity ~ 9.8 m/s/s

    ]
=#
p=[350,3,1.5,1.5,550,10000,3500,1.2,-0.5,9.8]

function diffeq_bicycle_model(du, u, p,t)
    bicycle_model!(du, u, p, com(t))
end



#=
#Check that this works vv

tspan = (0.0, 20.0)
prob=ODEProblem(diffeq_bicycle_model,u0,tspan,p) #doesn't set dt and uses adaptive time steping
sol = solve(prob)

#plot path of car
plot(sol, vars=(1,2))
=# 

#Alternative method for generation generation

#=
"""
    percent_noise_flat(x, percent_noise)

Computes noise by adding or subtracting a specified percentage of the provided
x array as error: noisy_x = x +/- (percent)*x
"""
function percent_noise_flat(x, percent_noise)
    # Create array of +/-1 the size of x
    noise = rand([Float64(-1.0),Float64(1.0)],length(x))
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
    noisy_x = Vector{Float64}(undef,length(x))
    # For each x, sample from the N(x, percent*x) distribution
    for i in 1:length(x)
        noisy_x[i] = x[i] * (Float64(1.0) + rand(Normal(0.0,percent_noise)))
    end
    return noisy_x
end

#test_xs[1,:]
#percent_noise_normal(test_xs[1,:],0.05)
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
#collect data
tspan = (0.0, 0.5)

data=[]
for i in 1:10
    println(i)
    function com(t)
        [5,0.1] .* 2 .*(rand(Float64, (2)).-0.5)
    end
    p=[350,3,1.5,1.5,550,10000,3500,1.2,-0.5,9.8].* 2 .*(rand(Float64, (length(p))).-0.5) #wrong, some of the params are linked to eachother & g=9.8
    u0=[0.0,0.0,0.0,5.0,0.0,0.0,0.0,0.0,0.0].+ 2 .*(rand(Float64, (length(u0))).-0.5)

    prob=ODEProblem(diffeq_bicycle_model,u0,tspan,p,dt=0.001,saveat=0.1) 
    sol = solve(prob)
    push!(data,[p,u0,sol.t, sol.u])
end