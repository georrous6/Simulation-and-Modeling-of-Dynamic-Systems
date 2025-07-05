function [best_gamma, best_M, best_sigma_bar] = tune_hyperparameters(train_input_fn, ...
    test_inputs, t, x0, f, basis, params, gamma_vals, M_vals, sigma_bar_vals)
% TUNE_HYPERPARAMETERS - Performs grid search to tune gamma, M, and sigma_bar
% for adaptive nonlinear model learning by minimizing prediction error on test inputs.
%
% Inputs:
%   train_input_fn   - Function handle for training input: @(t) -> scalar
%   test_inputs      - Cell array of function handles for test inputs
%   t                - Time vector
%   x0               - Initial condition for the system
%   f                - System dynamics function: f(t, x, u_handle)
%   basis            - Basis type used for regression ('poly', 'gauss', 'cos', etc.)
%   params           - Parameters specific to the selected basis
%   gamma_vals       - Array of candidate learning rates for adaptation
%   M_vals           - Array of candidate M values for sigma-modification
%   sigma_bar_vals   - Array of candidate maximum sigma values
%
% Outputs:
%   best_gamma       - Optimal gamma value that minimized error
%   best_M           - Optimal M value for sigma projection
%   best_sigma_bar   - Optimal sigma_bar value for regularization

    % Determine dimensions of the search space
    n_gamma = length(gamma_vals);
    n_M = length(M_vals);
    n_sigma = length(sigma_bar_vals);
    n_test = length(test_inputs);
    dt = t(2) - t(1);

    % Generate training data from system
    u_train_data = train_input_fn(t);
    odefun_train = @(t, x) f(t, x, train_input_fn);
    [~, x_train] = ode45(odefun_train, t, x0);
    Phi_train = generate_regressor(x_train, u_train_data, basis, params);

    % Preallocate error matrix: gamma × M × sigma × test inputs
    errors = NaN(n_gamma, n_M, n_sigma, n_test);

    % Grid search over hyperparameters
    for g = 1:n_gamma
        for m = 1:n_M
            for s = 1:n_sigma

                gamma = gamma_vals(g);
                M = M_vals(m);
                sigma_bar = sigma_bar_vals(s);

                % Train model with current hyperparameters
                [~, theta_hist] = gradient_nonlinear(x_train, Phi_train, gamma, M, sigma_bar, dt);
                theta_hat = theta_hist(end, :)';  % Final parameter estimate

                % Test model on each test input
                for j = 1:n_test
                    u_test_fn = test_inputs{j};
                    u_test_data = u_test_fn(t);
                    odefun_test = @(t, x) f(t, x, u_test_fn);
                    [~, x_test] = ode45(odefun_test, t, x0);

                    % Generate test regressor and compute prediction
                    Phi_test = generate_regressor(x_test, u_test_data, basis, params);
                    x_hat_test = Phi_test * theta_hat;

                    % Compute and store prediction error
                    errors(g, m, s, j) = sum((x_test - x_hat_test).^2);
                end
            end
        end
    end

    % Aggregate error across all test inputs for each parameter combo
    total_errors = sum(errors, 4);  % Sum over test input dimension

    % Find indices of minimum total error
    [~, idx] = min(total_errors(:));
    [g_opt, m_opt, s_opt] = ind2sub(size(total_errors), idx);

    % Output best parameters
    best_gamma = gamma_vals(g_opt);
    best_M = M_vals(m_opt);
    best_sigma_bar = sigma_bar_vals(s_opt);

    % Display results
    fprintf('Best hyperparameters found:\n');
    fprintf('  gamma      = %.4f\n', best_gamma);
    fprintf('  M          = %d\n', best_M);
    fprintf('  sigma_bar  = %.4f\n', best_sigma_bar);
end
