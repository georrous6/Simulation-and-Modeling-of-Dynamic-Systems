function [model_params, model_order] = model_complexity_vs_error(basis_types, max_params, f, t, x0, inputs, M, sigma_bar, gamma, filename)
%MODEL_COMPLEXITY_VS_ERROR Evaluate model error versus complexity for different bases
%
%   [model_params, model_order] = MODEL_COMPLEXITY_VS_ERROR(basis_types, max_params, f, t, x0, inputs, M, sigma_bar, gamma, filename)
%   runs system simulations with different model complexities and basis functions,
%   estimates model parameters using adaptive gradient methods, and computes the mean squared error (MSE)
%   on both training and testing datasets.
%
%   Inputs:
%       basis_types : cell array of strings specifying basis functions to test
%                     options: 'poly', 'gauss', 'cos'
%       max_params  : maximum number of basis functions (model complexity)
%       f           : system dynamics function handle, f(t,x,u)
%       t           : time vector for simulation
%       x0          : initial state vector
%       inputs      : cell array with two input functions {u_train, u_test}, each u(t) returns input at time t
%       M           : upper bound parameter for sigma-modification in adaptive law
%       sigma_bar   : sigma modification gain (positive scalar)
%       gamma       : adaptation gain (positive scalar)
%       filename    : string filename to save the error plot (vector graphics format)
%
%   Outputs:
%       model_params : cell array with the optimal parameters struct for each basis type
%       model_order  : vector with the optimal number of basis functions for each basis type

    % Extract training input and simulate training data
    u1 = inputs{1};
    u_train = u1(t)';
    odefun = @(t, x) f(t, x, u1);
    [~, x_train] = ode45(odefun, t, x0);
    
    % Extract testing input and simulate testing data
    u2 = inputs{2};
    u_test = u2(t)';
    odefun = @(t, x) f(t, x, u2);
    [~, x_test] = ode45(odefun, t, x0);

    dt = t(2) - t(1);
    n_bases = length(basis_types);

    % Initialize outputs and storage
    model_order = NaN(n_bases, 1);
    model_params = cell(n_bases, 1);
    colors = lines(n_bases);  % distinct colors for plotting

    training_errors_all = NaN(n_bases, max_params);
    testing_errors_all = NaN(n_bases, max_params);
    params_all = cell(n_bases, max_params);
    
    % Loop over basis types and model complexities
    for b = 1:n_bases
        basis = basis_types{b};
    
        for i = 1:max_params
            % Setup basis-specific parameters
            switch basis
                case 'poly'
                    params = struct('order', i);
    
                case 'gauss'
                    % Gaussian centers linearly spaced over training data range
                    centers = linspace(min(x_train), max(x_train), i);
                    width = (max(x_train) - min(x_train)) / i;
                    params = struct('centers', centers, 'width', width);
    
                case 'cos'
                    freqs = 1:i;
                    params = struct('freqs', freqs);
    
                otherwise
                    error('Unsupported basis type "%s"', basis);
            end
    
            % Generate regressor matrix for training data
            Phi_train = generate_regressor(x_train, u_train, basis, params);

            % Estimate parameters using nonlinear gradient descent
            [x_hat_train, theta_hist] = gradient_nonlinear(x_train, Phi_train, gamma, M, sigma_bar, dt);

            % Compute training mean squared error (MSE)
            training_errors_all(b, i) = mean((x_train - x_hat_train).^2);
            params_all{b, i} = params;
    
            % Testing phase: apply estimated parameters to testing data
            theta_hat = theta_hist(end, :)';
            Phi_test = generate_regressor(x_test, u_test, basis, params);
            x_hat_test = Phi_test * theta_hat;

            % Compute testing MSE
            testing_errors_all(b, i) = mean((x_test - x_hat_test).^2);
        end

        % Find model order minimizing testing error for this basis type
        [~, model_order(b)] = min(testing_errors_all(b,:));
        fprintf('Optimal number of parameters for %s-basis model: %d\n', basis, model_order(b));

        % Store optimal parameters
        model_params{b} = params_all{b, model_order(b)};
    end
    
    % Plot training and testing errors versus model complexity
    figure; hold on;
    
    for b = 1:length(basis_types)
        plot(1:max_params, testing_errors_all(b,:), '-', 'LineWidth', 1.8, ...
            'Color', colors(b,:), 'DisplayName', [basis_types{b} ' (Test)']);
        plot(1:max_params, training_errors_all(b,:), '--', 'LineWidth', 1.2, ...
            'Color', colors(b,:), 'DisplayName', [basis_types{b} ' (Train)']);
    end
    
    xlabel('Number of Parameters');
    ylabel('MSE');
    title('Training & Testing Error vs Model Complexity');
    legend('Location', 'best');
    grid on;
    
    % Export figure to file in vector graphic format
    exportgraphics(gcf, filename, 'ContentType', 'vector');

end