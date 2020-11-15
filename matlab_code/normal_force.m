function [FzF, FzR] = normal_force(x,u,p)
    %Moment Balance on car
        %Traction due to aero
        vx = x(4); % Lon Velocity
        F_lift = 0.5*p.rho*p.cla*vx^2;
        %Split forces on front and back axles
        %Takes into account acceleration weight shift
        %FzF = -( p.lr*(p.mass*p.g + F_lift) - p.cg_h*p.mass*u(2) )/ p.l;
        %FzR = -( p.lf*(p.mass*p.g + F_lift) + p.cg_h*p.mass*u(2) )/ p.l;
        FzF = -( p.lr*(p.mass*p.g + F_lift))/ p.l;
        FzR = -( p.lf*(p.mass*p.g + F_lift) )/ p.l;
        
     %Outputs forces (positive downforce)

end