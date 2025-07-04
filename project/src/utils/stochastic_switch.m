function u = stochastic_switch(t_span, T_dwell)
    t0 = t_span(1);
    tf = t_span(end);
    t_switches = t0:T_dwell:tf;
    values = randi([0, 1], size(t_switches));
    u = @(tau) interp1(t_switches, values, tau, 'previous', 'extrap');
end