function [x_hat, theta_hist] = gradient_nonlinear(t, x_vals, u_vals, basis_fun, gamma, M, sigma_bar)

    % Assertions
    assert(M > 0, sprintf('Assertion M > 0 failed: M=%f\n', M));
    assert(gamma > 0, sprintf('Assertion gamma > 0 failed: gamma=%f\n', gamma));
    assert(sigma_bar > 0, sprintf('Assertion sigma_bar > 0 failed: sigma_bar=%f\n', sigma_bar));

    d = length(basis_fun(1,1));  % parameter vector size (number of basis functions)
    
    % Init
    n = length(t);              % Number of time points
    theta_hat = zeros(d,1);     % initial guess
    theta_hist = zeros(n, d);   % store estimates over time
    x_hat = zeros(n,1);         % estimated output
    dt = diff(t);
    
    % Online loop
    for i = 1:n-1
        phi = basis_fun(x_vals(i), u_vals(i))';  % current regressor vector
        x_hat(i) = theta_hat' * phi;             % prediction
        e = x_vals(i) - x_hat(i);                  % error
    
        % Sigma-modification
        norm_theta = norm(theta_hat);
        if norm_theta < M
            sigma = 0;
        elseif norm_theta <= 2*M
            sigma = sigma_bar * (norm_theta/M - 1);
        else
            sigma = sigma_bar;
        end
    
        % Update rule (Euler discretization of gradient law)
        theta_hat_dot = -gamma * sigma * theta_hat + gamma * e * phi;
        theta_hat = theta_hat + dt(i) * theta_hat_dot;
    
        % Store
        theta_hist(i,:) = theta_hat';
    end

    x_hat(end) = x_hat(end-1);
    theta_hist(end,:) = theta_hist(end-1,:);
end