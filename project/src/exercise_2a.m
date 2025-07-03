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
          @(t) exp(-t) + ones(size(t));
          @(t) sin(pi*t) + sin(3*pi*t)};

filename = fullfile(outputDir, 'task2_cross_validation.pdf');
model_cross_validation(basis_types, model_params, model_order, f, t, x0, inputs, M, sigma_bar, gamma, filename);

%% Final Model Selection
N = 5;
inputs = cell(1, N);

for k = 1:N
    inputs{k} = @(t) 0;
    for n = 1:k
        freq = n;
        prevFunc = inputs{k};
        inputs{k} = @(t) prevFunc(t) + sin(freq * pi * t);
    end
end

for i = 1:N
    u = inputs{i};
    u_train = u(t);
    odefun = @(t, x) f(t, x, u);
    [~, x_train] = ode45(odefun, t, x0);

    % Training phase
    Phi_train = generate_regressor(x_train, u_train, 'poly', model_params{1});
    [~, theta_hist] = gradient_nonlinear(x_train, Phi_train, gamma, M, sigma_bar, dt);
end
