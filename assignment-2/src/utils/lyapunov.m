function [m_hat, b_hat, k_hat, y_hat] = lyapunov(x, m_0, b_0, k_0, u, dt, type)

    N = size(x, 1);
    m_hat = NaN(N, 1);
    b_hat = NaN(N, 1);
    k_hat = NaN(N, 1);
    y_hat = NaN(N, 1);

    % Initial parameter guesses
    A = [0, 1; -k_0 / m_0, -b_0 / m_0];
    B = [0; 1 / m_0];
    x_hat = x(1,:)';  % initialize state estimate

    if strcmp(type, 'parallel')
        for i = 1:N

            % Error
            e = x(i,:)' - x_hat;

            % Update laws (Euler integration)
            A = A + dt * (x_hat * e');
            B = B + dt * (u(i) * e);

            % State estimate update
            x_hat = x_hat + dt * (A * x_hat + B * u(i));

            % Save parameter estimates
            m_hat(i) = 1 / B(2);
            b_hat(i) = -A(2,2) / B(2);
            k_hat(i) = -A(2,1) / B(2);
            y_hat(i) = x_hat(1);
        end
    end
end
