function [x_hat, theta_hist] = gradient_nonlinear(x_vals, Phi, gamma, M, sigma_bar, dt)

    % Assertions
    assert(M > 0, sprintf('Assertion M > 0 failed: M=%f\n', M));
    assert(gamma > 0, sprintf('Assertion gamma > 0 failed: gamma=%f\n', gamma));
    assert(sigma_bar > 0, sprintf('Assertion sigma_bar > 0 failed: sigma_bar=%f\n', sigma_bar));

    [n, d] = size(Phi);         % parameter vector size (number of basis functions)
    theta_hat = zeros(d, 1);    % initial guess
    theta_hist = zeros(n, d);   % store estimates over time
    x_hat = zeros(n,1);         % estimated output
    
    % Online loop
    for i = 1:n-1
        phi = Phi(i,:)';                         % current regressor vector
        x_hat(i) = theta_hat' * phi;             % prediction
        e = x_vals(i) - x_hat(i);                % error
    
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
        theta_hat = theta_hat + dt * theta_hat_dot;
    
        % Store
        theta_hist(i,:) = theta_hat';
    end

    x_hat(end) = x_hat(end-1);
    theta_hist(end,:) = theta_hist(end-1,:);
end
