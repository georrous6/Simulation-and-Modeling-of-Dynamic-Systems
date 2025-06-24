clc; clearvars; close all;

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
dt = t(2) - t(1);

% Nonlinear Dynamics
f = @(t, x, ufun) -x^3 + theta_true(1)*tanh(x) + theta_true(2)/(1 + x^2) + ufun(t);

% Get training data
u1 = @(t) ones(size(t));
u_train = u1(t)';
odefun = @(t, x) f(t, x, u1);
[~, x_train] = ode45(odefun, t, x0);

% Get testing data
u2 = @(t) sin(t);
u_test = u2(t)';
odefun = @(t, x) f(t, x, u2);
[~, x_test] = ode45(odefun, t, x0);

%% Online Estimation with Lyapunov Gradient + Sigma-Modification (Multiple Bases)

% Setup
gamma = 0.1;
M = 10;
sigma_bar = 5;
max_params = 16;

basis_types = {'poly', 'gauss', 'cos'};
colors = lines(length(basis_types));  % distinct plot colors

% Preallocate
training_errors_all = NaN(length(basis_types), max_params);
testing_errors_all = NaN(length(basis_types), max_params);

for b = 1:length(basis_types)
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

        % Testing phase
        theta_hat = theta_hist(end, :)';
        Phi_test = generate_regressor(x_test, u_test, basis, params);
        x_hat_test = Phi_test * theta_hat;
        testing_errors_all(b, i) = mean((x_test - x_hat_test).^2);
    end
end


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
filename = fullfile(outputDir, 'task2_modeling_error_vs_model_complexity.pdf');
exportgraphics(gcf, filename, 'ContentType', 'vector');

