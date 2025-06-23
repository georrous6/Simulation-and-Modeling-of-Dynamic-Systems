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
fs = 10000;
t = 0:1/fs:T;
dt = t(2) - t(1);

% Nonlinear Dynamics
f = @(t, x, ufun) -x^3 + theta_true(1)*tanh(x) + theta_true(2)/(1 + x^2) + ufun(t);

% Get training data
u1 = @(t) sin(t) + sin(3 * t);
u_train = u1(t)';
odefun = @(t, x) f(t, x, u1);
[~, x_train] = ode45(odefun, t, x0);

% Get testing data
u2 = @(t) sin(2 * t);
u_test = u2(t)';
odefun = @(t, x) f(t, x, u2);
[~, x_test] = ode45(odefun, t, x0);

%% Online Estimation with Lyapunov Gradient + Sigma-Modification

% Parameters
gamma = 0.1;                % learning rate
M = 10;                     % threshold for σ-modification
sigma_bar = 15;             % max sigma value
max_params = 14;            % Maximum number of parameters i.e. length of the regressor vector
basis = 'poly';
training_errors = NaN(1, max_params);
testing_errors = NaN(1, max_params);

for i = 1:max_params

    params = struct('order', i);

    % Estimate parameters from training data
    Phi = generate_regressor(x_train, u_train, 'poly', params);
    [x_hat_train, theta_hist] = gradient_nonlinear(x_train, Phi, gamma, M, sigma_bar, dt);

    % Compute bias error from training data
    training_errors(i) = mean((x_train - x_hat_train).^2);

    % Evaluate model from testing data
    theta_hat = theta_hist(end,:)';
    Phi = generate_regressor(x_test, u_test, 'poly', params);
    x_hat_test = Phi * theta_hat;

    % Compute modeling error from testing data
    testing_errors(i) = mean((x_test - x_hat_test).^2);
end

figure; hold on;
plot(1:max_params, testing_errors, '-k', 'LineWidth', 1.5);
plot(1:max_params, training_errors, '--r', 'LineWidth', 1);
legend({'Training Error', 'Testing Error'});
xlabel('Number of Parameters');
ylabel('Modeling Error');
title('Modeling Error vs Model Complexity');
grid on;
