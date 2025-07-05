function [best_u_train_handle, best_theta_hat] = training_input_tuning( ...
    n_max_freq, test_inputs, t, x0, f, basis, params, gamma, M, sigma_bar, filename)
% TRAINING_INPUT_TUNING - Selects the best training input signal to minimize prediction error
% across multiple test scenarios using a nonlinear adaptive model.
%
% Inputs:
%   n_max_freq     - Maximum number of sinusoidal frequencies for training input generation
%   test_inputs    - Cell array of test input function handles (@(t) -> scalar)
%   t              - Time vector for simulation
%   x0             - Initial condition for system state
%   f              - System dynamics function: f(t, x, u_handle)
%   basis          - Basis type ('poly', 'gauss', 'cos', etc.)
%   params         - Parameters for the selected basis
%   gamma          - Learning rate for parameter update
%   M              - Bound threshold for sigma-modification
%   sigma_bar      - Maximum value of sigma for regularization
%   filename       - Filename to export error plot (vector format)
%
% Outputs:
%   best_u_train_handle - Function handle for best-performing training input
%   best_theta_hat      - Estimated parameters from best training input

    dt = t(2) - t(1);  % Time step
    train_inputs = cell(1, n_max_freq);

    % Construct training input signals: sum of sinusoids up to k frequencies
    for k = 1:n_max_freq
        train_inputs{k} = @(t) sum(sin((1:k)' * pi * t)) + 1;
    end

    n_test = length(test_inputs);  % Number of test scenarios
    cum_squared_error = NaN(n_max_freq, n_test);
    theta_hats = cell(n_max_freq, 1);  % store all parameter estimates

    % Loop over each training input
    for i = 1:n_max_freq
        u1 = train_inputs{i};                 % Current training input function
        u_train = u1(t);                      % Evaluate input over time
        odefun = @(t, x) f(t, x, u1);         % System dynamics with training input
        [~, x_train] = ode45(odefun, t, x0);  % Simulate system to get output

        % Generate regressor and perform adaptive learning
        Phi_train = generate_regressor(x_train, u_train, basis, params);
        [~, theta_hist] = gradient_nonlinear(x_train, Phi_train, gamma, M, sigma_bar, dt);
        theta_hat = theta_hist(end, :)';  % Final parameter estimates
        theta_hats{i} = theta_hat;        % Store for later use

        % Evaluate on all test input scenarios
        for j = 1:n_test
            u2 = test_inputs{j};                 % Test input function
            u_test = u2(t);                      % Evaluate test input
            odefun = @(t, x) f(t, x, u2);        % System dynamics with test input
            [~, x_test] = ode45(odefun, t, x0);  % Simulate system

            % Generate test regressor and predict output
            Phi_test = generate_regressor(x_test, u_test, basis, params);
            x_hat_test = Phi_test * theta_hat;

            % Compute cumulative squared error
            cum_squared_error(i,j) = sum((x_test - x_hat_test).^2);
        end
    end

    % Sum total error across all test inputs for each training input
    total_error = sum(cum_squared_error, 2);

    % Select the best training input based on minimum total error
    [~, best_idx] = min(total_error);
    best_u_train_handle = train_inputs{best_idx};  % Best training input
    best_theta_hat = theta_hats{best_idx};         % Corresponding parameter estimates

    % Plot the cumulative error across frequencies for each test input
    figure;
    plot(1:n_max_freq, cum_squared_error, 'LineWidth', 1.5);
    xlabel('Number of Training Input Frequencies');
    ylabel('Total Error');
    title('Modeling Error vs Number of Frequencies on Training Input');
    legend(arrayfun(@(i) sprintf('Test %d', i), 1:n_test, 'UniformOutput', false));
    grid on;

    % Export plot to file
    exportgraphics(gcf, filename, 'ContentType', 'vector');
end
