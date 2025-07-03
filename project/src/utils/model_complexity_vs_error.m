function [model_params, model_order] = model_complexity_vs_error(basis_types, max_params, f, t, x0, inputs, M, sigma_bar, gamma, filename)

    % Get training data
    u1 = inputs{1};
    u_train = u1(t)';
    odefun = @(t, x) f(t, x, u1);
    [~, x_train] = ode45(odefun, t, x0);
    
    % Get testing data
    u2 = inputs{2};
    u_test = u2(t)';
    odefun = @(t, x) f(t, x, u2);
    [~, x_test] = ode45(odefun, t, x0);

    dt = t(2) - t(1);
    n_bases = length(basis_types);
    model_order = NaN(n_bases, 1);
    model_params = cell(n_bases, 1);
    colors = lines(n_bases);  % distinct plot colors

    % Preallocate
    training_errors_all = NaN(n_bases, max_params);
    testing_errors_all = NaN(n_bases, max_params);
    params_all = cell(n_bases, max_params);
    
    for b = 1:n_bases
        basis = basis_types{b};
    
        for i = 1:max_params
            switch basis
                case 'poly'
                    params = struct('order', i);
    
                case 'gauss'
                    % Use i Gaussian centers linearly spaced between min and max of training x
                    centers = linspace(min(x_train), max(x_train), i);
                    width = (max(x_train) - min(x_train)) / i;
                    params = struct('centers', centers, 'width', width);
    
                case 'cos'
                    freqs = 1:i;
                    params = struct('freqs', freqs);
    
                otherwise
                    error('Unsupported basis type "%s"', basis);
            end
    
            % Training phase
            Phi_train = generate_regressor(x_train, u_train, basis, params);
            [x_hat_train, theta_hist] = gradient_nonlinear(x_train, Phi_train, gamma, M, sigma_bar, dt);
            training_errors_all(b, i) = mean((x_train - x_hat_train).^2);
            params_all{b, i} = params;
    
            % Testing phase
            theta_hat = theta_hist(end, :)';
            Phi_test = generate_regressor(x_test, u_test, basis, params);
            x_hat_test = Phi_test * theta_hat;
            testing_errors_all(b, i) = mean((x_test - x_hat_test).^2);
        end

        [~, model_order(b)] = min(testing_errors_all(b,:));
        fprintf('Optimal number of parameters for %s-basis model: %d\n', basis, model_order(b));
        model_params{b} = params_all{b, model_order(b)};
    end
    
    % Plot Training and Testing Errors
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
    
    % Export file
    exportgraphics(gcf, filename, 'ContentType', 'vector');

end