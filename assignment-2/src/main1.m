clc, clearvars, close all;

addpath('utils\');
outputDir = fullfile('..', 'plot');

% Check if the output directory exists, if not, create it
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

m = 1.315;
b = 0.225;
k = 0.725;

m_0 = 1.0;
b_0 = 0.1;
k_0 = 0.5;

T = 20;
dt = 0.01;

t = (0:dt:T)';
N = length(t);

u1 = @(t) 2.5 * ones(size(t));
u2 = @(t) 2.5 * sin(t);
inputs = {u1, u2};
labels = {'2.5', '2.5*sin(t)'};

p1 = -0.1; % -0.1
p2 = -0.2; % -0.2
Lambda = [1, -(p1 + p2), p1 * p2];
gammas = [1e-5, 1e-3];
x0 = [0; 0];

T0 = 10;
alpha0 = 0.001;
alpha1 = 140;

%% Estimate parameters using Gradient Descend method
phi = NaN(N, 3);

for i = 1:length(inputs)

    u = inputs{i};
    odefun = @(t, x) ([0, 1; -k/m, -b/m] * x(:) + [0; 1/m] * u(t));
    [~, x] = ode45(odefun, t, x0);
    phi(:,1) = lsim(tf(1, Lambda), x(:,1), t(:));
    phi(:,2) = lsim(tf(1, Lambda), x(:,2), t(:));
    phi(:,3) = lsim(tf(1, Lambda), u(t), t(:));

    gamma = gammas(i);
    [m_hat, b_hat, k_hat, y_hat] = gradientDescend(x(:,1), phi, m_0, b_0, k_0, Lambda, gamma);
    
    figure;
    hold on; grid on;
    plot(t, [m_hat, b_hat, k_hat], 'LineWidth', 1);
    plot(t, repmat([m, b, k], N, 1), '--r', 'LineWidth', 1);
    legend({'$\hat{m}$', '$\hat{b}$', '$\hat{k}$'}, 'Interpreter', 'latex');
    xlabel('t');
    title(sprintf('Parameter estimations for input signal u(t)=%s (gamma=%f)', labels{i}, gamma));
    filePath = fullfile(outputDir, sprintf('task1_parameter_estimations_gradient_descend_%d.pdf', i));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');
    
    e = x(:,1) - y_hat;
    figure;
    hold on; grid on;
    plot(t, [x(:,1), y_hat, e], 'LineWidth', 1);
    legend({'$x$', '$\hat{x}$', 'e'}, 'Interpreter', 'latex');
    xlabel('t');
    title(sprintf('Identification error for input signal u(t)=%s (gamma=%f)', labels{i}, gamma));
    filePath = fullfile(outputDir, sprintf('task1_identification_error_gradient_descend_%d.pdf', i));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');
    
    figure;
    is_PE = persistenceOfExcitationCondition(phi, t, T0, alpha0, alpha1);
    filePath = fullfile(outputDir, sprintf('task1_PE_eigenvalues_%d.pdf', i));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');
    if is_PE
        fprintf('Persistence of Excitation Condition is satisfied for u(t)=%s\n', labels{i});
    else
        fprintf('Persistence of Excitation Condition is NOT satisfied for u(t)=%s\n', labels{i});
    end
end

%% Estimate parameters using Lyapunov (parallel and mixed) method

u = inputs{2};
odefun = @(t, x) ([0, 1; -k/m, -b/m] * x(:) + [0; 1/m] * u(t));
[~, x] = ode45(odefun, t, x0);
structures = {'parallel', 'mixed'};
A_real = [0, 1; -k/m, -b/m];
C = 100 * eye(2);
x_0 = [0; 0];

for i = 1:length(structures)
    [m_hat, b_hat, k_hat, y_hat, V_dot] = lyapunov(x, m_0, b_0, k_0, x_0, u(t), dt, structures{i}, A_real, C);

    figure;
    hold on; grid on;
    plot(t, V_dot, 'LineWidth', 1);
    xlabel('t');
    title(sprintf('Lyapunov (%s) derivative function', structures{i}));
    filePath = fullfile(outputDir, sprintf('task1_lyapunov_derivative_function_%s.pdf', structures{i}));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');

    figure;
    hold on; grid on;
    plot(t, [m_hat, b_hat, k_hat], 'LineWidth', 1);
    plot(t, repmat([m, b, k], N, 1), '--r', 'LineWidth', 1);
    legend({'$\hat{m}$', '$\hat{b}$', '$\hat{k}$'}, 'Interpreter', 'latex');
    xlabel('t');
    title(sprintf('Lyapunov (%s): Parameter estimations for u(t)=%s', structures{i}, labels{2}));
    filePath = fullfile(outputDir, sprintf('task1_parameter_estimations_lyapunov_%s.pdf', structures{i}));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');
    
    e = x(:,1) - y_hat;
    figure;
    hold on; grid on;
    plot(t, [x(:,1), y_hat, e], 'LineWidth', 1);
    legend({'$x$', '$\hat{x}$', 'e'}, 'Interpreter', 'latex');
    xlabel('t');
    title(sprintf('Lyapunov (%s): Identification error for u(t)=%s', structures{i}, labels{2}));
    filePath = fullfile(outputDir, sprintf('task1_identification_error_lyapunov_%s.pdf', structures{i}));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');
end

%% Estimate parameters using Lyapunov (parallel and mixed) method with noise

eta0 = 0.25;
f0 = 20;
eta = eta0 * sin(2 * pi * f0 * t);
x_noise = [eta + x(:,1), x(:,2)];

for i = 1:length(structures)

    [m_hat, b_hat, k_hat, y_hat] = lyapunov(x, m_0, b_0, k_0, x_0, u(t), dt, structures{i}, A_real, C);
    [m_hat_noise, b_hat_noise, k_hat_noise, y_hat_noise] = lyapunov(x_noise, m_0, b_0, k_0, x_0, u(t), dt, structures{i}, A_real, C);

    figure;
    hold on; grid on;
    plot(t, abs(x(:,1) - y_hat), 'LineWidth', 1);
    plot(t, abs(x(:,1) - y_hat_noise), 'LineWidth', 1);

    legend({'without noise', 'with noise'}, 'Interpreter', 'latex');
    xlabel('t'); ylabel('|e|');
    title(sprintf('Lyapunov (%s): Identification error for u(t)=%s', structures{i}, labels{2}));
    filePath = fullfile(outputDir, sprintf('task1_identification_error_lyapunov_%s_noise.pdf', structures{i}));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');
end

eta0_values = linspace(0.1, 1, 10);
n = length(eta0_values);
m_tilde = NaN(n, 1);
b_tilde = NaN(n, 1);
k_tilde = NaN(n, 1);

for i = 1:length(structures)

    for j = 1:n
        eta = eta0_values(j) * sin(2 * pi * f0 * t);
        x_noise = [eta + x(:,1), x(:,2)];
        [m_hat_noise, b_hat_noise, k_hat_noise, y_hat_noise] = lyapunov(x_noise, m_0, b_0, k_0, x_0, u(t), dt, structures{i}, A_real, C);

        m_tilde(j) = abs(m - m_hat_noise(end));
        b_tilde(j) = abs(b - b_hat_noise(end));
        k_tilde(j) = abs(k - k_hat_noise(end));
    end

    figure;
    hold on; grid on;
    plot(eta0_values, [m_tilde, b_tilde, k_tilde], 'LineWidth', 1);

    legend({'$|\tilde{m}|$', '$|\tilde{b}|$', '$|\tilde{k}|$'}, 'Interpreter', 'latex');
    xlabel('$\eta_0$', 'Interpreter', 'latex');
    title(sprintf('Lyapunov (%s): Parameter estimation error vs Noise Amplitude', structures{i}));
    filePath = fullfile(outputDir, sprintf('task1_estimation_parameter_error_vs_noise_amplitude_lyapunov_%s.pdf', structures{i}));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');
end
