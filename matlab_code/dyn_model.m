function [x_dot] = dyn_model(x,u)
%u = [v_theta, accel, steer_target]
%x = [x, y, psi, vx, vy, r, theta, steer_angle]

  %Create param Object
  p = param;

  %Unpack vehicle state
  psi  = x(3); %heading
  u_x  = x(4); %velocity x
  u_y  = x(5); %velocity y
  r    = x(6); %yawrate
  steer = x(8); %current steering angle
  v_theta = u(1); %velocity along path
  D = u(2); %accel command
  delta = u(3); %commanded steer rate

  %Compute slip angles
  alpha_f = atan((u_y + r*p.lf)/u_x) + steer;
  alpha_r = atan((u_y - r*p.lr)/u_x);

  %Estimate Normal
  [FzF, FzR] = normal_force(x,u,p);
  
  %Compute tire forces
  F_yf = tire_force(alpha_f, FzF, p);
  F_yr = tire_force(alpha_r, FzR, p);

  %Torque to force
  F_net = p.mass*D;

  %Torque Vectoring
  F_xf = p.lf / p.l * F_net;
  F_xr = p.lr / p.l * F_net;

  % Acceleration
  a_x = 1/p.mass*(F_xr + F_xf*cos(steer) + F_yf*sin(steer)) + r*u_y;
  
  a_y = 1/p.mass*(F_yr - F_xf*sin(steer) + F_yf*cos(steer)) - r*u_x;
  
  a_yaw = 1/p.Iz*(-p.lf*F_xf*sin(steer) + p.lf*F_yf*cos(steer) - p.lr*F_yr);

  %Standard Dynamics
  x_dot_1 =   u_x*cos(psi) - u_y*sin(psi);
  x_dot_2 =   u_x*sin(psi) + u_y*cos(psi);
  x_dot_3 =   r;
  x_dot_4 =   a_x;
  x_dot_5 =   a_y;
  x_dot_6 =   a_yaw;

  %Velocity theta
  x_dot_7 = v_theta;
  
  %Steering
  x_dot_8 = delta;

  x_dot = [x_dot_1 x_dot_2 x_dot_3 x_dot_4 x_dot_5 x_dot_6 x_dot_7 x_dot_8]';
end
