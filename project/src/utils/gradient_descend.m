function Y = gradient_descend(t, x, u, lambda, gamma, M, sigma_bar, omegafun, A, B, A_0, B_0)

    theta1_star = [lambda + A(1,1); A(1,2); B(1)];
    theta2_star = [A(2,1); lambda + A(2,2); B(2)];

    max_theta_star_norm = max(norm(theta1_star), norm(theta2_star));

    % Assertions
    assert(M >= max_theta_star_norm, sprintf('Assertion M >= %f failed: M=%f\n', ...
        max_theta_star_norm, M));
    assert(lambda > 0, sprintf('Assertion lambda > 0 failed: lambda=%f\n', lambda));
    assert(gamma > 0, sprintf('Assertion gamma > 0 failed: gamma=%f\n', gamma));
    assert(sigma_bar > 0, sprintf('Assertion sigma_bar > 0 failed: sigma_bar=%f\n', sigma_bar));

    % Unpack state
    t = t(:);
    u = u(:);
    n = length(t);
    Y = NaN(n, 10);

    theta1_hat = [lambda + A_0(1,1); A_0(1,2); B_0(1)];
    theta2_hat = [A_0(2,1); lambda + A_0(2,2); B_0(2)];

    % Regressor dynamics
    phi = NaN(n, 3);
    phi(:,1) = lsim(tf(1, [1, lambda]), x(:,1), t);
    phi(:,2) = lsim(tf(1, [1, lambda]), x(:,2), t);
    phi(:,3) = lsim(tf(1, [1, lambda]), u, t);

    % Bias terms
    omega = omegafun(t);

    % True outputs (with bias)
    x1 = phi * theta1_star + omega(:,1);
    x2 = phi * theta2_star + omega(:,2);

    Y(:,1:2) = [x1, x2];
    Y(1,3:4) = [0, 0];
    Y(1,5:end) = [theta1_hat', theta2_hat'];
    dt = diff(t);

    for i = 1:n-1

        % Estimated outputs
        x1_hat = theta1_hat' * phi(i,:)';
        x2_hat = theta2_hat' * phi(i,:)';
    
        % Errors
        e1 = x1(i) - x1_hat;
        e2 = x2(i) - x2_hat;
    
        % Sigma modifications
        sigma1 = sigma_switch(theta1_hat, M, sigma_bar);
        sigma2 = sigma_switch(theta2_hat, M, sigma_bar);
    
        % Adaptive laws
        theta1_hat_dot = -gamma * sigma1 * theta1_hat + gamma * e1 * phi(i,:)';
        theta2_hat_dot = -gamma * sigma2 * theta2_hat + gamma * e2 * phi(i,:)';

        theta1_hat = theta1_hat + dt(i) * theta1_hat_dot;
        theta2_hat = theta2_hat + dt(i) * theta2_hat_dot;

        Y(i+1,3:4) = [x1_hat, x2_hat];
        Y(i+1,5:end) = [theta1_hat(1) - lambda, theta1_hat(2), ...  % a11, a12
                        theta2_hat(1), theta2_hat(2) - lambda, ...  % a21, a22
                        theta1_hat(3), theta2_hat(3)];              % b1, b2
    end
end