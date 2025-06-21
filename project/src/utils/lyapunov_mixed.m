function [Y, V] = lyapunov_mixed(y0, u, t, A, B, C, Gamma)

    [~, Y] = ode45(@(t, y) lyapunov_ode(t, y, u, A, B, C, Gamma), t, y0);
    n = length(t);
    V = NaN(n, 2);
    
    % Find Lyapunov function values
    for i = 1:n
        A_hat = [Y(i,5), Y(i,6); Y(i,7), Y(i,8)];
        B_hat = Y(i, 9:10)';
        A_tilde = A_hat - A;
        B_tilde = B_hat - B;
        e = Y(i,1:2)' - Y(i,3:4)';
        V(i,1) = (e' * e + trace(A_tilde' * A_tilde) + trace(B_tilde' * B_tilde)) / 2;
    end

    V(1:end-1,2) = diff(V(:,1)) ./ diff(t(:));
end