function [x_hat, theta_hist] = gradient_nonlinear(x_vals, Phi, gamma, M, sigma_bar, dt)
%GRADIENT_NONLINEAR Online parameter estimation using sigma-modified gradient descent
%
%   [x_hat, theta_hist] = GRADIENT_NONLINEAR(x_vals, Phi, gamma, M, sigma_bar, dt)
%   estimates parameters of a nonlinear regression model online using
%   gradient descent with sigma-modification to ensure parameter boundedness.
%
%   Inputs:
%       x_vals    - Nx1 vector of measured outputs (data to fit)
%       Phi       - NxD regressor matrix, each row corresponds to regression vector at time i
%       gamma     - positive scalar, adaptation gain
%       M         - positive scalar, sigma-modification threshold parameter
%       sigma_bar - positive scalar, sigma-modification gain
%       dt        - scalar time step size (sampling interval)
%
%   Outputs:
%       x_hat     - Nx1 vector of estimated outputs (predictions)
%       theta_hist- NxD matrix of parameter estimates over time

    % Verify positive parameters
    assert(M > 0, sprintf('Assertion M > 0 failed: M=%f\n', M));
    assert(gamma > 0, sprintf('Assertion gamma > 0 failed: gamma=%f\n', gamma));
    assert(sigma_bar > 0, sprintf('Assertion sigma_bar > 0 failed: sigma_bar=%f\n', sigma_bar));

    [n, d] = size(Phi);         % n: number of samples, d: number of parameters
    theta_hat = zeros(d, 1);    % initialize parameter estimates
    theta_hist = zeros(n, d);   % history of parameter estimates
    x_hat = zeros(n,1);         % initialize output estimates
    
    for i = 1:n-1
        phi = Phi(i,:)';                         % current regressor vector (column)
        x_hat(i) = theta_hat' * phi;             % predicted output at time i
        e = x_vals(i) - x_hat(i);                % prediction error
    
        % Sigma-modification: compute gain to prevent parameter drift
        norm_theta = norm(theta_hat);
        if norm_theta < M
            sigma = 0;
        elseif norm_theta <= 2*M
            sigma = sigma_bar * (norm_theta/M - 1);
        else
            sigma = sigma_bar;
        end
    
        % Parameter update: Euler discretization of gradient adaptive law
        theta_hat_dot = -gamma * sigma * theta_hat + gamma * e * phi;
        theta_hat = theta_hat + dt * theta_hat_dot;
    
        % Store parameter estimates
        theta_hist(i,:) = theta_hat';
    end

    % For last time step, copy previous estimates to avoid NaNs
    x_hat(end) = x_hat(end-1);
    theta_hist(end,:) = theta_hist(end-1,:);
end
