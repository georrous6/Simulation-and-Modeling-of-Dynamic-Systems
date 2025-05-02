function dz = lyapunov_nonlinear(t, z, u, C, theta_star, d)
    % Extract state
    x = z(1:2);
    hat_x = z(3:4);
    theta1_hat = z(5:7);
    theta2_hat = [1; z(9)];
    theta1_star = theta_star(1:3);
    theta2_star = theta_star(4:5);

    % f(x) and g(x)
    f = [0, 0, 0;
        -x(2), -sin(x(1)), x(2)^2 * sin(2*x(1))];

    g = [x(2), 0;
         0, u(t, x)];

    % True system dynamics
    dx = f * theta1_star + g * theta2_star + [0; 1] * d(t);

    % Estimated dynamics
    hat_dx = f * theta1_hat + g * theta2_hat + C * (x - hat_x);

    % Error
    e = x - hat_x;

    % Adaptation laws
    dtheta1_hat = f' * e;
    dtheta2_hat = g' * e;

    % Output derivative
    dz = [dx;
          hat_dx;
          dtheta1_hat;
          dtheta2_hat];
end