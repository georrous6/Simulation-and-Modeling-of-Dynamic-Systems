function [t_out, x] = simulateSystemResponse(m, L, c, g, A0, w, times)

    % Write state-space equations
    A = [0, 1; -g / L, -c / (m * L^2)];
    B = [0; 1 / (m * L^2)];

    % Solve the system using ode45 function
    u = @(t) (A0 * sin(w * t));
    odefun = @(t, x) (A * x + B * u(t));
    [t_out, x] = ode45(odefun, times, [0; 0]);
end