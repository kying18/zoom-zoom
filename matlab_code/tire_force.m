function F = tire_force(alpha,Fz,p)

F = -alpha .* p.cornering_stiff .* (Fz/p.sample_fz);

end