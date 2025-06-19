function [theta_hat, V] = lyapunov_mixed(theta_0, x_val, u_val, t, C, A, B)
    theta_0 = theta_0(:);

    x_hat = theta_0(1:2);
    A_hat = reshape(theta_0(3:6), [2, 2]);
    B_hat = theta_0(7:8);

    N = size(x_val, 1);
    theta_hat = zeros(N, length(theta_0));
    theta_hat(1,:) = theta_0';
    V = NaN(N, 2);

    dt = diff(t);
   
    for i = 2:N
        x = x_val(i,:)';
        u = u_val(i);
        e = x - x_hat;

        x_hat_dot = A_hat * x + B_hat * u + C * e;
        A_hat_dot = e * x';
        B_hat_dot = e * u';

        % Construct 6x1 derivative vector
        theta_dot = [A_hat_dot(:); B_hat_dot];
        
        % Construct 6x1 parameter vector
        theta = [A_hat(:); B_hat];
        
        % Apply projection
        theta_dot_projected = project_theta_dot(theta_dot, theta);
        
        % Unpack projected derivatives
        A_hat_dot = reshape(theta_dot_projected(1:4), [2, 2]);
        B_hat_dot = theta_dot_projected(5:6);
        
        % Update parameter estimates
        x_hat = x_hat + dt(i-1) * x_hat_dot;
        A_hat = A_hat + dt(i-1) * A_hat_dot;
        B_hat = B_hat + dt(i-1) * B_hat_dot;

        % Find Lyapunov function value along with its derivative
        A_tilde = A_hat - A;
        B_tilde = B_hat - B;
        V(i,1) = (e' * e + trace(A_tilde' * A_tilde) + trace(B_tilde' * B_tilde)) / 2;
        V(i,2) = -e' * C * e;
        
        theta_hat(i,1:2) = x_hat';
        theta_hat(i,3:6) = A_hat(:);
        theta_hat(i,7:8) = B_hat';
    end

end