function [m_hat, b_hat, k_hat, y_hat, V_dot] = lyapunov(x, m_0, b_0, k_0, x_0, u, dt, structure, A_real, C)

    N = size(x, 1);
    m_hat = NaN(N, 1);
    b_hat = NaN(N, 1);
    k_hat = NaN(N, 1);
    y_hat = NaN(N, 1);
    V_dot = NaN(N, 1);

    % Initial parameter guesses
    A_hat = [0, 1; -k_0 / m_0, -b_0 / m_0];
    B_hat = [0; 1 / m_0];
    x_hat = x_0;  % initialize state estimate

    if strcmp(structure, 'parallel')
        for i = 1:N

            % Error
            e = x(i,:)' - x_hat;

            V_dot(i) = e' * A_real * e;

            % Update laws (Euler integration)
            A_hat = A_hat + dt * (x_hat * e');
            B_hat = B_hat + dt * (u(i) * e);

            % State estimate update
            x_hat = x_hat + dt * (A_hat * x_hat + B_hat * u(i));

            % Save parameter estimates
            m_hat(i) = 1 / B_hat(2);
            b_hat(i) = -A_hat(2,2) / B_hat(2);
            k_hat(i) = -A_hat(2,1) / B_hat(2);
            y_hat(i) = x_hat(1);
        end
    elseif strcmp(structure, 'mixed')

        % Check if matrix C is possitive semi-definite
        if ~issymmetric(C) || any(eig(C) < -1e-8)
            error('Matrix C must be symetric and positive semi-definite');
        end

        for i = 1:N

            % Error
            e = x(i,:)' - x_hat;

            V_dot(i) = -e' * C * e;

            % Update laws (Euler integration)
            A_hat = A_hat + dt * (x(i,:)' * e');
            B_hat = B_hat + dt * (u(i) * e);

            % State estimate update
            x_hat = x_hat + dt * (A_hat * x(i,:)' + B_hat * u(i) + C * e);

            % Save parameter estimates
            m_hat(i) = 1 / B_hat(2);
            b_hat(i) = -A_hat(2,2) / B_hat(2);
            k_hat(i) = -A_hat(2,1) / B_hat(2);
            y_hat(i) = x_hat(1);
        end
    else
        error('Invalid Lyapunov structure');
    end
end
