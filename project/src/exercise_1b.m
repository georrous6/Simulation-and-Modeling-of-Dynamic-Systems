clc, clearvars, close all;

addpath('utils');

outputDir = fullfile('..', 'plot');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Define system matrices
A = [-2.15, 0.25; -0.75, -2];
B = [0; 1.5];

% Define sampling frequency
fs = 20;

% Initial state
x0 = [0; 0];

% Define time points
T = 100;                     % Simulation time
t = 0:1/fs:T;

lambda = 5;                 % Filter parameter
fw = 0.01;                  % Bias signal frequency (<< fn)
omega_bar = 0.1;            % Bias amplitude
M = 10;                     % sigma-modification threshold
sigma_bar = 5;              % Maximum sigma damping
gamma = 10;                 % Adaptation gain

% Initial parameter guesses
A_0 = [-2, 0.5; -0.5, -1.5];
B_0 = [0; 1];

% Input signal
u = @(t) 2 * sin(t);

% Bias error
omegafun = @(t) omega_bar * [sin(2*pi*fw*t(:)), cos(2*pi*fw*t(:))];

odefun = @(t, x) A * x + B * u(t);
[~, x] = ode45(odefun, t, x0);
Y = gradient_descend(t, x, u(t), lambda, gamma, M, sigma_bar, omegafun, A, B, A_0, B_0);


%% Plot Parameter Estimations
figure;
plot(t, Y(:,5:end), 'LineWidth', 1.5);
legend({'$\hat{\alpha}_{11}$', '$\hat{\alpha}_{12}$', ...
        '$\hat{\alpha}_{21}$', '$\hat{\alpha}_{22}$', ...
        '$\hat{b}_1$', '$\hat{b}_2$'}, 'Interpreter', 'latex');
grid on;
xlabel('t [sec]');
title('Parameter Estimations of Gradient Descend');

filename = fullfile(outputDir, 'task1_gradient_parameter_estimations.pdf');
exportgraphics(gcf, filename, 'ContentType', 'vector');


%% Plot Identification Error
n_states = size(x, 2);
x_measured = Y(:,1:2);
x_hat = Y(:,3:4);
e = x_measured - x_hat;

figure('Position', [200, 100, 800, 400]);
sgtitle('Identification Error');
for j = 1:n_states
    subplot(1, 2, j);
    plot(t, [x_measured(:,j), x_hat(:,j), e(:,j)], 'LineWidth', 1.5);
    legend({sprintf('$x_%d$', j), ...
            sprintf('$\\hat{x}_%d$', j), ...
            sprintf('$e_%d$', j)}, 'Interpreter', 'latex');
    grid on;
    xlabel('t [sec]');
    title(sprintf('State %d', j));
end

filename = fullfile(outputDir, 'task1_gradient_identification_error.pdf');
exportgraphics(gcf, filename, 'ContentType', 'vector');


%% Estimation MSE vs Bias Amplitude
n_values = 20;
omega_bar_values = linspace(0.1, 10, n_values);
mse_theta_values = NaN(n_values, 6);
mse_state_values = NaN(n_values, 2);
theta_star = [A(1,1), A(1,2), A(2,1), A(2,2), B(1), B(2)];
for i = 1:n_values
    omega_bar = omega_bar_values(i);
    omegafun = @(t) omega_bar * [sin(2*pi*fw*t(:)), cos(2*pi*fw*t(:))];
    Y = gradient_descend(t, x, u(t), lambda, gamma, M, sigma_bar, omegafun, A, B, A_0, B_0);
    x_measured = Y(:,1:2);
    x_hat = Y(:,3:4);
    theta_hat = Y(:,5:end);
    mse_theta_values(i,:) = mean((theta_hat - theta_star).^2);
    mse_state_values(i,:) = mean((x_measured - x_hat).^2);
end

% Plot parameter estimation error vs bias amplitude
figure;
plot(omega_bar_values, mse_theta_values, 'LineWidth', 1.5);
xlabel('$\bar{\omega}$', 'Interpreter', 'latex');
ylabel('Parametric MSE');
title('Parametric MSE vs Bias Amplitude');
grid on;

filename = fullfile(outputDir, 'task1_parametric_mse_vs_bias_amplitude.pdf');
exportgraphics(gcf, filename, 'ContentType', 'vector');

% Plot identification error vs bias amplitude
figure;
plot(omega_bar_values, mse_state_values, 'LineWidth', 1.5);
xlabel('$\bar{\omega}$', 'Interpreter', 'latex');
ylabel('Identification MSE');
title('Identification MSE vs Bias Amplitude');
grid on;

filename = fullfile(outputDir, 'task1_identification_mse_vs_bias_amplitude.pdf');
exportgraphics(gcf, filename, 'ContentType', 'vector');
