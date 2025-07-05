function [Y, V] = lyapunov_mixed(y0, u, t, A, B, C, Gamma)
%LYAPUNOV_MIXED Computes system trajectories and Lyapunov function for adaptive system
%
%   [Y, V] = LYAPUNOV_MIXED(y0, u, t, A, B, C, Gamma) integrates the system
%   dynamics using ode45 and evaluates a mixed Lyapunov function over time.
%
%   Inputs:
%       y0    - initial state vector (including states and parameters)
%       u     - input signal vector, sampled at times t
%       t     - time vector for integration
%       A, B  - true system parameter matrices
%       C     - output matrix (not used in this function, passed to ODE)
%       Gamma - adaptation gain matrix for parameter updates
%
%   Outputs:
%       Y     - matrix of system states and parameter estimates over time
%       V     - Nx2 matrix of Lyapunov function values and their time derivatives
%               V(:,1) is the Lyapunov function value at each time
%               V(:,2) is its numerical time derivative (dV/dt)

    % Solve system ODE using ode45 with nested function lyapunov_ode
    [~, Y] = ode45(@(t, y) lyapunov_ode(t, y, u, A, B, C, Gamma), t, y0);

    n = length(t);
    V = NaN(n, 2);  % preallocate Lyapunov function and derivative
    
    for i = 1:n
        % Extract estimated parameters from solution Y
        A_hat = [Y(i,5), Y(i,6); Y(i,7), Y(i,8)];
        B_hat = Y(i, 9:10)';

        % Parameter estimation errors
        A_tilde = A_hat - A;
        B_tilde = B_hat - B;

        % State estimation error (actual - estimat
        e = Y(i,1:2)' - Y(i,3:4)';

        % Lyapunov function (scalar)
        V(i,1) = (e' * e + trace(A_tilde' * A_tilde) + trace(B_tilde' * B_tilde)) / 2;
    end

    % Numerical time derivative of Lyapunov function using finite differences
    V(1:end-1,2) = diff(V(:,1)) ./ diff(t(:));
end