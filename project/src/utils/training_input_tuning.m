function [best_u_train_handle, best_theta_hat] = training_input_tuning( ...
    n_max_freq, test_inputs, t, x0, f, basis, params, gamma, M, sigma_bar, filename)

    dt = t(2) - t(1);
    train_inputs = cell(1, n_max_freq);
    for k = 1:n_max_freq
        train_inputs{k} = @(t) sum(sin((1:k)' * pi * t)) + 1;
    end

    n_test = length(test_inputs);
    cum_squared_error = NaN(n_max_freq, n_test);
    theta_hats = cell(n_max_freq, 1);  % store all parameter estimates

    % Loop over each training input
    for i = 1:n_max_freq
        u1 = train_inputs{i};
        u_train = u1(t);
        odefun = @(t, x) f(t, x, u1);
        [~, x_train] = ode45(odefun, t, x0);

        Phi_train = generate_regressor(x_train, u_train, basis, params);
        [~, theta_hist] = gradient_nonlinear(x_train, Phi_train, gamma, M, sigma_bar, dt);
        theta_hat = theta_hist(end, :)';
        theta_hats{i} = theta_hat;  % save for later

        % Evaluate on all test inputs
        for j = 1:n_test
            u2 = test_inputs{j};
            u_test = u2(t);
            odefun = @(t, x) f(t, x, u2);
            [~, x_test] = ode45(odefun, t, x0);   

            Phi_test = generate_regressor(x_test, u_test, basis, params);
            x_hat_test = Phi_test * theta_hat;

            cum_squared_error(i,j) = sum((x_test - x_hat_test).^2);
        end
    end

    % Sum over test inputs for each training input
    total_error = sum(cum_squared_error, 2);

    % Find best input
    [~, best_idx] = min(total_error);
    best_u_train_handle = train_inputs{best_idx};
    best_theta_hat = theta_hats{best_idx};  % corresponding parameter estimates

    % Optional: plot results
    figure;
    plot(1:n_max_freq, cum_squared_error, 'LineWidth', 1.5);
    xlabel('Number of Training Input Frequencies');
    ylabel('Total Error');
    title('Modeling Error vs Number of Frequencies on Training Input');
    legend(arrayfun(@(i) sprintf('Test %d', i), 1:n_test, 'UniformOutput', false));
    grid on;

    % Export file
    exportgraphics(gcf, filename, 'ContentType', 'vector');
end
