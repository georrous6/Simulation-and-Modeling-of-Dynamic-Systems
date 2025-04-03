function [c_hat, L_hat, m_hat] = estimateParameters(u_s, x_s, t_s, outputDerivativeAvailable, p1, p2, g)

    N = length(t_s);  % Number of samples
    phi = zeros(N, 3);

    phi(:,2) = lsim(tf(-1, [1, -(p1 + p2), p1 * p2]), x_s(:,1), t_s);
    phi(:,3) = lsim(tf(1, [1, -(p1 + p2), p1 * p2]), u_s, t_s);
    if outputDerivativeAvailable
        phi(:,1) = lsim(tf(-1, [1, -(p1 + p2), p1 * p2]), x_s(:,2), t_s);
    else
        phi(:,1) = lsim(tf([-1, 0], [1, -(p1 + p2), p1 * p2]), x_s(:,1), t_s);
    end

    % Estimate parameters using least squares method
    theta = (phi' * phi) \ (phi' * x_s(:,1));
    c_hat = (theta(1) - p1 - p2) / theta(3);
    L_hat = g / (p1 * p2 + theta(2));
    m_hat = 1 / (theta(3) * L_hat^2);
end