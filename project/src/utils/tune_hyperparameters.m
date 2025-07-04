function [best_gamma, best_M, best_sigma_bar] = tune_hyperparameters(train_input_fn, ...
    test_inputs, t, x0, f, basis, params, gamma_vals, M_vals, sigma_bar_vals)

    % Dimensions
    n_gamma = length(gamma_vals);
    n_M = length(M_vals);
    n_sigma = length(sigma_bar_vals);
    n_test = length(test_inputs);
    dt = t(2) - t(1);

    % Generate training data
    u_train_data = train_input_fn(t);
    odefun_train = @(t, x) f(t, x, train_input_fn);
    [~, x_train] = ode45(odefun_train, t, x0);
    Phi_train = generate_regressor(x_train, u_train_data, basis, params);

    % Initialize error storage
    errors = NaN(n_gamma, n_M, n_sigma, n_test);

    % Grid search over hyperparameters
    for g = 1:n_gamma
        for m = 1:n_M
            for s = 1:n_sigma

                gamma = gamma_vals(g);
                M = M_vals(m);
                sigma_bar = sigma_bar_vals(s);

                % Train
                [~, theta_hist] = gradient_nonlinear(x_train, Phi_train, gamma, M, sigma_bar, dt);
                theta_hat = theta_hist(end, :)';

                % Test on each input
                for j = 1:n_test
                    u_test_fn = test_inputs{j};
                    u_test_data = u_test_fn(t);
                    odefun_test = @(t, x) f(t, x, u_test_fn);
                    [~, x_test] = ode45(odefun_test, t, x0);

                    Phi_test = generate_regressor(x_test, u_test_data, basis, params);
                    x_hat_test = Phi_test * theta_hat;

                    % Store individual error
                    errors(g, m, s, j) = sum((x_test - x_hat_test).^2);
                end
            end
        end
    end

    % Compute total error across all test inputs
    total_errors = sum(errors, 4);

    % Find best parameters
    [~, idx] = min(total_errors(:));
    [g_opt, m_opt, s_opt] = ind2sub(size(total_errors), idx);

    best_gamma = gamma_vals(g_opt);
    best_M = M_vals(m_opt);
    best_sigma_bar = sigma_bar_vals(s_opt);

    % Optional display
    fprintf('Best hyperparameters found:\n');
    fprintf('  gamma      = %.4f\n', best_gamma);
    fprintf('  M          = %d\n', best_M);
    fprintf('  sigma_bar  = %.4f\n', best_sigma_bar);
end
