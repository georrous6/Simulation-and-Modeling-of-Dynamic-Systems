function y_dot = lyapunov_ode(t, y, u, A, B, C, Gamma)
    x = y(1:2);
    x_hat = y(3:4);
    e = x - x_hat;
    A_hat = [y(5), y(6); y(7), y(8)];
    B_hat = y(9:10);

    x_dot = A * x + B * u(t);
    x_hat_dot = A_hat * x + B_hat * u(t) + C * e;
    A_hat_dot = e * x';
    B_hat_dot = e * u(t)';

    theta_dot = [A_hat_dot(1,1); A_hat_dot(1,2); A_hat_dot(2,1); A_hat_dot(2,2); B_hat_dot];

    % Project data
    if nargin > 6
        theta_dot = projection(theta_dot, y(5:10), Gamma);
    end

    y_dot = [x_dot; x_hat_dot; theta_dot];
end