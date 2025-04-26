function is_PE = persistenceOfExcitationCondition(phi, t, T0, alpha0, alpha1)

    is_PE = true;
    dt = t(2) - t(1);
    N = size(phi, 1);
    L = round(T0 / dt);    % number of samples per window

    if T0 <= dt
        error('T0 must be greater that dt');
    end
    
    eigenvalues = NaN(N-L, 3);
    
    for k = 1:N-L
        phi_sum = zeros(3,3);
        for j = k:k+L-1
            phi_sum = phi_sum + phi(j,:)' * phi(j,:) * dt;
        end
        
        % Check eigenvalues
        eigVals = eig(phi_sum);
        eigenvalues(k,:) = eigVals;
        
        if any(eigVals > alpha1 * T0) || any(eigVals < alpha0 * T0)
            is_PE = false;
        end
    end

    hold on; grid on;
    plot(t(1:N-L), eigenvalues, 'LineWidth', 1);
    plot(t(1:N-L), T0 * [alpha1, alpha0] .* ones(N-L, 2), '--r', 'LineWidth', 1);
    xlabel('t');
    title(sprintf('Persistence of Excitation Eigenvalues ($$T_0=%.1f, \\, \\alpha_0 = %.1f,\\, \\alpha_1=%.1f$$)', T0, alpha0, alpha1), 'Interpreter', 'latex');
    legend({'$\lambda_1$', '$\lambda_2$', '$\lambda_3$'}, 'Interpreter', 'latex');
    hold off;
end
