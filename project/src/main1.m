clc, clearvars, close all;

outputDir = fullfile('..', 'plot');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Define system matrices
A = [-2.15, 0.25; -0.75, -2];
B = [0; 1.5];

% Define time span
tspan = [0, 10];

% Initial condition
x0 = [0; 0];

% Define input u(t) as function handle
u1 = @(t) 1;

% Define ODE function
odefun = @(t, x) A * x + B * u1(t);

% Solve using ode45
[t, x] = ode45(odefun, tspan, x0);

% Plot results
figure;
plot(t, x(:,1), 'r-', 'LineWidth', 1.5); hold on;
plot(t, x(:,2), 'b-', 'LineWidth', 1.5);
xlabel('t [sec]');
legend('x_1(t)', 'x_2(t)');
title('System Step Response');
grid on;

filename = fullfile(outputDir, 'task1_step_response.pdf');
exportgraphics(gcf, filename, 'ContentType', 'vector');
