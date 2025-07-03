function kfold_avg_mse = model_cross_validation(basis_types, model_params, order, f, t, x0, inputs, M, sigma_bar, gamma, filename)

    K = length(inputs);
    N = length(t);
    n = K * N;
    u_all = zeros(n, 1);
    x_all = zeros(n, 1);
    kfold_indices = zeros(n, 1);

    for i = 1:K
        u = inputs{i};
        idx = (i - 1) * N + 1:i * N;
        u_all(idx) = u(t);
        odefun = @(t, x) f(t, x, u);
        [~, x_all(idx)] = ode45(odefun, t, x0);
        kfold_indices(idx) = i;
    end

    dt = t(2) - t(1);
    n_bases = length(basis_types);
    colors = lines(n_bases);  % distinct plot colors
    kfold_avg_mse = NaN(n_bases, K);
    
    for b = 1:n_bases
        basis = basis_types{b};
        params = model_params{b};
    
        for i = 1:K

            cum_mse = 0;
            u_train = u_all(kfold_indices == i);
            x_train = x_all(kfold_indices == i);

            % Training phase
            Phi_train = generate_regressor(x_train, u_train, basis, params);
            [~, theta_hist] = gradient_nonlinear(x_train, Phi_train, gamma, M, sigma_bar, dt);
            
            for j = 1:K
                if j ~= i
                    u_test = u_all(kfold_indices == j);
                    x_test = x_all(kfold_indices == j);

                    % Testing phase
                    theta_hat = theta_hist(end, :)';
                    Phi_test = generate_regressor(x_test, u_test, basis, params);
                    x_hat_test = Phi_test * theta_hat;
                    cum_mse = cum_mse + mean((x_test - x_hat_test).^2);
                end
            end

            kfold_avg_mse(b, i) = cum_mse / (K-1);
        end
    end
    
    % Plot Training and Testing Errors
    figure; hold on;
    
    for b = 1:length(basis_types)
        plot(1:K, kfold_avg_mse(b,:), '-o', 'LineWidth', 1.8, 'Color', ...
            colors(b,:), 'DisplayName', sprintf('%s(%d)', basis_types{b}, order(b)));
    end
    
    xlabel('Train Fold Index');
    ylabel('Average K-Fold MSE');
    title('K-Fold Cross Validation');
    legend('Location', 'best');
    grid on;
    
    % Export file
    exportgraphics(gcf, filename, 'ContentType', 'vector');

end
