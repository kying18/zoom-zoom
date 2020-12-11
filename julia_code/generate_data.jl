cd(@__DIR__)
using Pkg
Pkg.activate(".")

# data generation
    # add noise
    # change parameters
    # save states, inputs, parameters as data
# david will write data generator
    # write code that modifies it to change parameters
        # figure out which parameters change
    # optional arguments

# use the Dormand Prince to get next time step

function update_parameters(p)
    pass
end

#### should be covered by David ####
function generate_inputs()
    pass
end

# Define the coefficients for the Dormand Prince method
c = [0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0]
b = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]
a2 = 1.0/5.0
a3 = [3.0/40.0, 9.0/40.0]
a4 = [44.0/45.0, -56.0/15.0, 32.0/9.0]
a5 = [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0]
a6 = [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0]
a7 = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0]

function dormandprince(f,x,u,p,dt)
    # Calculate stages 1-6
    k1 = f(x,u,p)
    k2 = f(muladd(dt,a2*k1,x),u,p)
    k3 = f(muladd(dt,muladd(a3[1],k1,a3[2]*k2),x),u,p)
    k4 = f(muladd(dt,muladd(a4[1],k1,muladd(a4[2],k2,a4[3]*k3)),x),u,p)
    k5 = f(muladd(dt,muladd(a5[1],k1,muladd(a5[2],k2,muladd(a5[3],k3,a5[4]*k4))),x),u,p)
    k6 = f(muladd(dt,muladd(a6[1],k1,muladd(a6[2],k2,muladd(a6[3],k3,muladd(a6[4],k4,a6[5]*k5)))),x),u,p)
    # The last step is only needed for error estimation
    # Calculate the next u
    x_next = muladd(dt,muladd(b[1],k1,muladd(b[3],k3,muladd(b[4],k4,muladd(b[5],k5,b[6]*k6)))),x),u,p)
    # Return the next u
    out = x_next
end

function run_dormand_prince(f,x0,u0,p0,t0,tf,dt)
    # Set up the time range
    t_range = t0:dt:tf
    # Figure out how long the time range is
    n = length(t_range)
    # Create vectors to save the data in
    x = Vector{typeof(x0)}(undef,n)
    u = Vector{typeof(u0)}(undef,n)
    p = Vector{typeof(p0)}(undef,n)
    # Assign the initial values to the first row
    x[1] = x0
    u[1] = u0
    p[1] = p0
    # Calculate the subsequent values using the Dormand Prince method
    for i in 1:n-1
        # Generate controls
        # Update the parameters
        # Calculate the next state of the vehicle with the new controls/parameters
        x[i+1] = dormandprince(f,x[i],u[i],p[i],dt)
    end
    [x u p]
end
