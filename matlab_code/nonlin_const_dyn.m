function [const] = nonlin_const_dyn(z)
p = param;

%Unpack values
u_x   = z(4); %velocity x
u_y   = z(5); %velocity y
r     = z(6); %yawrate
delta = z(8); %current steering angle
a_x   = z(10);%commanded accel

%Find Lateral Forces
x = z(1:8);
u = z(9:11);

%Compute slip angles
alpha_f = atan((u_y + r*p.lf)/u_x) + delta;
alpha_r = atan((u_y - r*p.lr)/u_x);

%Compute Normal Forces
[FzF, FzR] = normal_force(x, u, p);

%Compute tire forces
F_Ry = tire_force(alpha_r, FzR, p);

%Longitudinal Forces (2 wheel drive)
F_Rx = a_x*p.mass; %force on back

%Friction Ellipse
R_R = F_Rx^2/(p.max_mu_lon*FzR)^2 + F_Ry^2/(p.max_mu_lat*FzR)^2 ;

%Pack as column vector
const = [R_R; alpha_f; alpha_r];

end
