clc; clearvars; close all;

%% Parameters and Nonlinear System Setup
theta_true = [1.2; 0.8];
x0 = 0;
T = 100;
dt = 0.01;
t = 0:dt:T;
n = length(t);

% Input
u = @(t) sin(t);
u_vals = u(t)';

% Simulate true nonlinear system
f = @(t, x, ufun) -x^3 + theta_true(1)*tanh(x) + theta_true(2)/(1 + x^2) + ufun(t);
odefun = @(t, x) f(t, x, u);
[~, x] = ode45(odefun, t, x0);

% Data: inputs and output (x = y here)
y = x;

%% Model Selection Pipeline

% models = {
%     struct('name', 'Linear (x, u)', 'basis', @(x,u) [x, u]);
%     struct('name', 'Quadratic (x, x², u)', 'basis', @(x,u) [x, x.^2, u]);
%     struct('name', 'Cubic (x, x², x³, u)', 'basis', @(x,u) [x, x.^2, x.^3, u])
% };
% 
% K = 5;  % number of CV folds
% cv_indices = kfold_indices(n, K);
% 
% results = struct();
% 
% for m = 1:length(models)
%     model = models{m};
%     mse_val = zeros(K,1);
% 
%     for k = 1:K
%         % Cross-validation split
%         val_idx = (cv_indices == k);
%         train_idx = ~val_idx;
% 
%         % Training data
%         Phi_train = model.basis(x(train_idx), u_vals(train_idx));
%         y_train = y(train_idx);
% 
%         % Validation data
%         Phi_val = model.basis(x(val_idx), u_vals(val_idx));
%         y_val = y(val_idx);
% 
%         % Estimate θ with least-squares
%         theta_hat = (Phi_train' * Phi_train) \ (Phi_train' * y_train);
% 
%         % Predict on validation set
%         y_pred = Phi_val * theta_hat;
% 
%         % MSE on validation
%         mse_val(k) = mean((y_pred - y_val).^2);
%     end
% 
%     results(m).name = model.name;
%     results(m).mean_mse = mean(mse_val);
%     results(m).std_mse = std(mse_val);
% end
% 
% %% Display Results
% fprintf('Model Evaluation Results:\n');
% for m = 1:length(results)
%     fprintf('%-35s  MSE = %.5f (±%.5f)\n', ...
%         results(m).name, results(m).mean_mse, results(m).std_mse);
% end
% 
% %% Plot Predictions of All Models vs True Output
% 
% figure;
% hold on;
% colors = lines(length(models));  % distinct colors
% 
% % Plot true output
% plot(t, y, 'k', 'LineWidth', 2, 'DisplayName', 'True Output');
% 
% for m = 1:length(models)
%     model = models{m};
% 
%     % Train using all data
%     Phi_full = model.basis(x, u_vals);
%     theta_hat = (Phi_full' * Phi_full) \ (Phi_full' * y);
%     y_hat = Phi_full * theta_hat;
% 
%     % Plot prediction
%     plot(t, y_hat, '--', 'Color', colors(m,:), 'LineWidth', 1.5, ...
%         'DisplayName', model.name);
% end
% 
% xlabel('Time [s]');
% ylabel('Output');
% title('Model Outputs vs True System Output');
% legend('Location', 'best');
% grid on;
% 
% %% Residuals Plot
% 
% figure;
% tiledlayout(length(models), 1, 'TileSpacing', 'compact');
% 
% for m = 1:length(models)
%     model = models{m};
% 
%     % Train using all data
%     Phi_full = model.basis(x, u_vals);
%     theta_hat = (Phi_full' * Phi_full) \ (Phi_full' * y);
%     y_hat = Phi_full * theta_hat;
% 
%     % Compute residual
%     residual = y - y_hat;
% 
%     % Plot residual
%     nexttile;
%     plot(t, residual, 'LineWidth', 1.2);
%     yline(0, '--k');
%     grid on;
%     title(sprintf('Residual: %s', model.name));
%     xlabel('Time [s]');
%     ylabel('e(t)');
% end


%% Online Estimation with Lyapunov Gradient + σ-Modification

% Choose model
basis_fun = @(x,u) [-x.^3, tanh(x), 1./(1 + x.^2), u];  % same basis as in the nonlinear model

% Parameters
gamma = 1;                 % learning rate
M = 10;                     % threshold for σ-modification
sigma_bar = 15;             % max sigma value

[x_hat, theta_hist] = gradient_nonlinear(t, x, u_vals, basis_fun, gamma, M, sigma_bar);

%% Plot: True vs Estimated Output
figure;
plot(t, x, 'k', 'LineWidth', 2); hold on;
plot(t, x_hat, 'r--', 'LineWidth', 1.5);
legend('True Output', 'Estimated Output');
xlabel('Time [s]'); ylabel('y(t)');
title('Online Model Output vs True Output');
grid on;

%% Plot: Residual
figure;
plot(t, x - x_hat, 'LineWidth', 1.2);
xlabel('Time [s]'); ylabel('Residual e(t)');
title('Residual over Time');
grid on;

%% Plot: Parameter Convergence
figure;
plot(t, theta_hist, 'LineWidth', 1.4);
xlabel('Time [s]'); ylabel('\thetâ_i(t)');
legend('\thetâ_1', '\thetâ_2', '\thetâ_3');
title('Online Parameter Estimates');
grid on;


