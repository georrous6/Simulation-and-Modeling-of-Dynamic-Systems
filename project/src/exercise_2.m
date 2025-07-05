clc; clearvars; close all;

rng(1);         % For reproducibility
addpath('utils');

outputDir = fullfile('..', 'plot');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Parameters and Nonlinear System Setup
theta_true = [1.2; 0.8];
x0 = 0;
T = 10;
fs = 1000;
t = 0:1/fs:T;

% Nonlinear Dynamics
f = @(t, x, ufun) -x^3 + theta_true(1)*tanh(x) + theta_true(2)/(1 + x^2) + ufun(t);

%% Model Structure Selection using Lyapunov Gradient + Sigma-Modification

% Setup
gamma = 0.1;
M = 10;
sigma_bar = 5;
max_params = 16;

basis_types = {'poly', 'gauss', 'cos'};
filename = fullfile(outputDir, 'task2_modeling_error_vs_model_complexity.pdf');

u1 = @(t) ones(size(t));
u2 = @(t) sin(t);
inputs = {u1, u2};

[model_params, model_order] = model_complexity_vs_error(basis_types, ...
    max_params, f, t, x0, inputs, M, sigma_bar, gamma, filename);

%% Candidate Models Cross Validation

% Generate input signals
inputs = {@(t) ones(size(t));
          @(t) sin(2*pi*t);
          @(t) sin(2*pi*t + sin(0.5*pi*t));
          @(t) exp(-t) .* sin(2*pi*t);
          @(t) 1 + exp(-t)};

filename = fullfile(outputDir, 'task2_cross_validation.pdf');
kfold_avg_mse = model_cross_validation(basis_types, model_params, model_order, ...
    f, t, x0, inputs, M, sigma_bar, gamma, filename);

[~, idx] = min(mean(kfold_avg_mse, 2));
basis = basis_types{idx};
params = model_params{idx};

%% Final Model Selection

% Select training input
n_max_freq = 10;
filename = fullfile(outputDir, 'task2_error_vs_training_input_complexity.pdf');
[u, theta_hat] = training_input_tuning(n_max_freq, inputs, t, x0, f, basis, ...
    params, gamma, M, sigma_bar, filename);

fprintf('\nFinal Model Parameters (%s-basis):\n', basis);
disp(theta_hat);

% Define hyperparameter grids
gamma_vals = [0.5, 1, 5, 10];
M_vals = [5, 10, 20, 50];
sigma_bar_vals = [0.001, 0.01, 0.1, 1];

% Hyperparameter tuning
[gamma, M, sigma_bar] = tune_hyperparameters(u, inputs, t, x0, f, basis, params, ...
    gamma_vals, M_vals, sigma_bar_vals);

%% Test Model Over Stochastic Input

% === Gaussian noise signal ===
gaussian_signal = randn(size(t));
u_gaussian = @(tau) interp1(t, gaussian_signal, tau, 'linear', 'extrap');

% === Stochastic binary signal with T_dwell ===
T_dwell = 0.5;
u_binary = stochastic_switch([t(1), t(end)], T_dwell);

% === Evaluate both test inputs ===
test_inputs = {u_gaussian, u_binary};
input_labels = {'Gaussian Noise', 'Stochastic Binary Signal'};
colors = {'b', 'm'};
lineWidths = [1, 2];

for i = 1:length(test_inputs)
    u_test = test_inputs{i};
    u_test_data = u_test(t);
    
    % Solve system with test input
    odefun_test = @(t, x) f(t, x, u_test);
    [~, x_test] = ode45(odefun_test, t, x0);
    
    % Prediction using model
    Phi_test = generate_regressor(x_test, u_test_data, basis, params);
    x_hat_test = Phi_test * theta_hat;

    % Create new figure per input
    figure;

    % Subplot 1: Input signal
    subplot(2,1,1);
    plot(t, u_test_data, 'LineWidth', lineWidths(i), 'Color', colors{i});
    xlabel('Time');
    ylabel('Input Signal');
    title(['Input Signal: ', input_labels{i}]);
    grid on;

    % Subplot 2: True vs predicted output
    subplot(2,1,2);
    plot(t, x_test, 'k', 'LineWidth', lineWidths(i)); hold on;
    plot(t, x_hat_test, [colors{i} '--'], 'LineWidth', lineWidths(i));
    legend('True Output', 'Predicted Output', 'Location', 'southeast');
    xlabel('Time');
    ylabel('System Output');
    title(['Model Response to ', input_labels{i}]);
    grid on;

    filename = fullfile(outputDir, sprintf('task2_final_model_vs_%s.pdf', ...
        lower(strrep(input_labels{i}, ' ', '_')))); 
    exportgraphics(gcf, filename, 'ContentType', 'vector');
end
