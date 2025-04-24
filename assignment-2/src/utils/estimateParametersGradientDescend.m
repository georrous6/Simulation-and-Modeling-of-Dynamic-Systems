function [m_hat, b_hat, k_hat, y_hat] = estimateParametersGradientDescend(x, u, t, m_0, b_0, k_0, Lambda, gamma)

    N = length(t);
    m_hat = NaN(N, 1);
    b_hat = NaN(N, 1);
    k_hat = NaN(N, 1);
    y_hat = NaN(N, 1);


    lambda1 = Lambda(2);
    lambda2 = Lambda(3);

    phi = NaN(N, 3);
    phi(:,1) = lsim(tf(1, Lambda), x(:,1), t(:));
    phi(:,2) = lsim(tf(1, Lambda), x(:,2), t(:));
    phi(:,3) = lsim(tf(1, Lambda), u(:), t(:));

    theta_hat = [lambda2 - k_0 / m_0, lambda1 - b_0 / m_0, 1 / m_0];
    for k = 1:N-1
        y_hat(k) = theta_hat * phi(k,:)';
        e = x(k,1) - y_hat(k);
        theta_hat_dot = gamma * e * phi(k,:);

        m_hat(k) = 1 / theta_hat(3);
        b_hat(k) = (lambda1 - theta_hat(2)) / theta_hat(3);
        k_hat(k) = (lambda2 - theta_hat(1)) / theta_hat(3);

        theta_hat = gamma * theta_hat_dot + theta_hat;
    end

end