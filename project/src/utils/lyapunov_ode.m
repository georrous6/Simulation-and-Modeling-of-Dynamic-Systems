function y_dot = lyapunov_ode(t, y, u, A, B, C, Gamma)
%LYAPUNOV_ODE Computes the state derivatives for adaptive system dynamics
%
%   y_dot = LYAPUNOV_ODE(t, y, u, A, B, C, Gamma) returns the time derivative
%   of the combined state and parameter vector y at time t.
%
%   Inputs:
%       t     - current time (scalar)
%       y     - current state vector, including:
%               x (1:2)       : actual system states
%               x_hat (3:4)   : estimated system states
%               A_hat (5:8)   : estimated system matrix parameters (2x2)
%               B_hat (9:10)  : estimated input parameters (2x1)
%       u     - function handle or vector for system input, u(t) returns input at time t
%       A, B  - true system matrices
%       C     - observer gain matrix for estimation error feedback
%       Gamma - adaptation gain matrix for projection (optional)
%
%   Output:
%       y_dot - time derivative of y (column vector)

    % Extract actual and estimated states
    x = y(1:2);
    x_hat = y(3:4);

    % Estimation error between actual and estimated states
    e = x - x_hat;

    % Extract estimated parameters A_hat and B_hat from y
    A_hat = [y(5), y(6); y(7), y(8)];
    B_hat = y(9:10);

    % System dynamics for actual states
    x_dot = A * x + B * u(t);

    % Estimated system dynamics with correction from estimation error
    x_hat_dot = A_hat * x + B_hat * u(t) + C * e;

    % Parameter adaptation laws (gradient-type updates)
    A_hat_dot = e * x';     % 2x2 matrix update for A_hat
    B_hat_dot = e * u(t)';  % 2x1 vector update for B_hat

    % Combine parameter derivatives into a vector
    theta_dot = [A_hat_dot(1,1); A_hat_dot(1,2); A_hat_dot(2,1); A_hat_dot(2,2); B_hat_dot];

    % Apply projection operator if Gamma is provided to keep parameters bounded
    if nargin > 6
        theta_dot = projection(theta_dot, y(5:10), Gamma);
    end

    % Combine derivatives into output vector
    y_dot = [x_dot; x_hat_dot; theta_dot];
end