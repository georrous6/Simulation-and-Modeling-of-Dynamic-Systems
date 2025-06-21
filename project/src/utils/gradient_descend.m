function y = gradient_descend(t, x, u, lambda, gamma, M, sigma_bar, omegafun, A, B, A_0, B_0)

    % Unpack state
    t = t(:);
    u = u(:);
    n = length(t);
    y = NaN(n, 10);

    theta1_star = [lambda + A(1,1); A(1,2); B(1)];
    theta2_star = [A(2,1); lambda + A(2,2); B(2)];

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

    y(:,1:2) = [x1, x2];
    x1_hat = x1(1);
    x2_hat = x2(1);
    y(1,3:4) = [x1_hat, x2_hat];
    y(1,5:end) = [theta1_hat', theta2_hat'];
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

        y(i+1,3:4) = [x1_hat, x2_hat];
        y(i+1,5:end) = [theta1_hat', theta2_hat'];
    end

    % Final output: [x1, x2, x1_hat, x2_hat, a11_hat, a12_hat, a21_hat, a22_hat, b1_hat, b2_hat]
    y = [y(:,1:4), y(:,5) - lambda, y(:,6), y(:,8), y(:,9) - lambda, y(:,7), y(:,10)];
end