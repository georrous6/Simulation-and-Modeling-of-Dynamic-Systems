function kfold_avg_mse = model_cross_validation(basis_types, model_params, order, f, t, x0, inputs, M, sigma_bar, gamma, filename)
%MODEL_CROSS_VALIDATION Perform K-fold cross-validation for model evaluation
%
%   kfold_avg_mse = MODEL_CROSS_VALIDATION(basis_types, model_params, order, f, t, x0, inputs, M, sigma_bar, gamma, filename)
%   performs K-fold cross-validation to evaluate the average mean squared error (MSE)
%   of models built with different basis functions and parameters.
%
%   Inputs:
%       basis_types  : cell array of strings specifying basis function types
%       model_params : cell array of parameter structs corresponding to each basis
%       order        : vector of model orders for each basis type
%       f            : system dynamics function handle f(t,x,u)
%       t            : time vector for simulation
%       x0           : initial state vector
%       inputs       : cell array of input functions for each fold (length K)
%       M            : upper bound parameter for sigma-modification in adaptive law
%       sigma_bar    : sigma modification gain (positive scalar)
%       gamma        : adaptation gain (positive scalar)
%       filename     : string filename to save the cross-validation plot
%
%   Output:
%       kfold_avg_mse : matrix of average K-fold MSE values
%                       size: [number of basis types x number of folds]

    K = length(inputs);   % Number of folds
    N = length(t);        % Number of time points per fold
    n = K * N;            % Total data length across all folds

    % Preallocate arrays for inputs, outputs, and fold indices
    u_all = zeros(n, 1);
    x_all = zeros(n, 1);
    kfold_indices = zeros(n, 1);

    % Concatenate data from all folds
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
    
    % Loop over each basis type
    for b = 1:n_bases
        basis = basis_types{b};
        params = model_params{b};
    
        % Perform K-fold cross-validation
        for i = 1:K

            cum_mse = 0;

            % Training data for fold i
            u_train = u_all(kfold_indices == i);
            x_train = x_all(kfold_indices == i);

            % Generate regressor matrix for training
            Phi_train = generate_regressor(x_train, u_train, basis, params);

            % Estimate parameters using nonlinear gradient descent
            [~, theta_hist] = gradient_nonlinear(x_train, Phi_train, gamma, M, sigma_bar, dt);
            
            % Test on all other folds (j ~= i)
            for j = 1:K
                if j ~= i
                    u_test = u_all(kfold_indices == j);
                    x_test = x_all(kfold_indices == j);

                    % Generate regressor matrix for testing fold
                    theta_hat = theta_hist(end, :)';
                    Phi_test = generate_regressor(x_test, u_test, basis, params);

                    % Predict outputs and compute MSE
                    x_hat_test = Phi_test * theta_hat;
                    cum_mse = cum_mse + mean((x_test - x_hat_test).^2);
                end
            end

            % Average MSE over the (K-1) testing folds
            kfold_avg_mse(b, i) = cum_mse / (K-1);
        end
    end
    
    % Plot average K-fold MSE for each basis type and fold
    figure; hold on;
    
    for b = 1:length(basis_types)
        plot(1:K, kfold_avg_mse(b,:), '-o', 'LineWidth', 1.8, 'Color', ...
            colors(b,:), 'DisplayName', sprintf('%s(%d)', basis_types{b}, order(b)));
    end
    
    xlabel('Train Fold Index');
    ylabel('Average K-Fold MSE');
    title('K-Fold Cross Validation');
    legend show;
    grid on;
    
    % Export figure to file
    exportgraphics(gcf, filename, 'ContentType', 'vector');

end
