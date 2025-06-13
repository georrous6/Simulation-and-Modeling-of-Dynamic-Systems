clc, clearvars, close all;

outputDir = fullfile('..', 'plot');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Define system matrices
A = [-2.15, 0.25; -0.75, -2];
B = [0; 1.5];

% Define sampling frequency
fs = 20;

% Define time points
t = 0:1/fs:15;

% Initial condition
x0 = [0; 0];

% Define input u(t) as function handle
u1 = @(t) ones(size(t));

% Define ODE function
odefun = @(t, x) A * x + B * u1(t);

% Solve using ode45
[~, x] = ode45(odefun, t, x0);

% Plot results
figure;
plot(t, x, 'LineWidth', 1.5);
xlabel('t [sec]');
legend({'$x_1$', '$x_2$'}, 'Interpreter', 'latex');
title('System Step Response');
grid on;

filename = fullfile(outputDir, 'task1_step_response.pdf');
exportgraphics(gcf, filename, 'ContentType', 'vector');

A_0 = [-2, 0; 1, 3];
B_0 = [0; 2];
C = eye(2);
theta_0 = [x0; A_0(:); B_0];
theta_hat = lyapunov_mixed(theta_0, x, u1(t), t, C);

figure;
plot(t, theta_hat(:,3:end), 'LineWidth', 1.5);
legend({'$\hat{\alpha}_{11}$', '$\hat{\alpha}_{21}$', ...
        '$\hat{\alpha}_{12}$', '$\hat{\alpha}_{22}$', ...
        '$\hat{b}_1$', '$\hat{b}_2$'}, 'Interpreter', 'latex');
grid on;
xlabel('t [sec]');
title('Parameter Estimations');
filename = fullfile(outputDir, 'task1_parameter_estimations.pdf');
exportgraphics(gcf, filename, 'ContentType', 'vector');

n_states = size(x, 2);
for i = 1:n_states
    x_hat = theta_hat(:,1:2);
    e = x - x_hat;
    figure;
    plot(t, [x(:,i), x_hat(:,i), e(:,i)], 'LineWidth', 1.5);
    legend({sprintf('$x_%d$', i), ...
            sprintf('$\\hat{x}_%d$', i), ...
            sprintf('$e_%d$', i)}, 'Interpreter', 'latex');
    grid on;
    xlabel('t [sec]');
    title(sprintf('Output Error (State %d)', i));

    filename = fullfile(outputDir, sprintf('task1_state_%d_error.pdf', i));
    exportgraphics(gcf, filename, 'ContentType', 'vector');
end

