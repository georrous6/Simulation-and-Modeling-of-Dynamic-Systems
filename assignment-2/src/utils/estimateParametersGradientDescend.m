function [m_hat, b_hat, k_hat, y_hat] = estimateParametersGradientDescend(y, phi, m_0, b_0, k_0, Lambda, gamma)

    N = length(y);
    m_hat = NaN(N, 1);
    b_hat = NaN(N, 1);
    k_hat = NaN(N, 1);
    y_hat = NaN(N, 1);

    if length(Lambda) == 3

        lambda1 = Lambda(2);
        lambda2 = Lambda(3);
    
        theta_hat = [lambda2 - k_0 / m_0, lambda1 - b_0 / m_0, 1 / m_0];
        for k = 1:N
            y_hat(k) = theta_hat * phi(k,:)';
            e = y(k) - y_hat(k);
    
            m_hat(k) = 1 / theta_hat(3);
            b_hat(k) = (lambda1 - theta_hat(2)) / theta_hat(3);
            k_hat(k) = (lambda2 - theta_hat(1)) / theta_hat(3);
    
            theta_hat = gamma * e * phi(k,:) + theta_hat;
        end
    end
end