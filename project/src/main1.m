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
t = 0:1/fs:30;

% Initial condition
x0 = [0; 0];

% Initial parameter estimations
A_0 = [-1.1, 1;
        -0.5, -4];
B_0 = [2.5; 2];
C = eye(2);
theta_0 = [x0; A_0(1,1); A_0(1,2); A_0(2,1); A_0(2,2); B_0];

% Input signals
u1 = @(t) ones(size(t));
u2 = @(t) 2 * sin(t);
inputs = {u1, u2};
inputNames = {'Step', 'Sinusoidal'};

fprintf('Parameter Estimations\n');
fprintf('===================\n');

for i = 1:length(inputs)
    %% Sovle ODE System

    % Select the input signal
    u = inputs{i};

    % Define ODE function
    odefun = @(t, x) A * x + B * u(t);

    % Solve using ode45
    [~, x] = ode45(odefun, t, x0);

    % Plot System Response
    figure;
    plot(t, x, 'LineWidth', 1.5);
    xlabel('t [sec]');
    legend({'$x_1$', '$x_2$'}, 'Interpreter', 'latex');
    title(sprintf('System %s Response', inputNames{i}));
    grid on;
    
    % Export files
    filename = fullfile(outputDir, sprintf('task1_%s_response.pdf', lower(inputNames{i})));
    exportgraphics(gcf, filename, 'ContentType', 'vector');
    
    %% Estimate Parameters
    [theta_hat, V] = lyapunov_mixed(theta_0, x, u(t), t, C, A, B);
    fprintf('%s Input:\n', inputNames{i});
    disp(theta_hat(end,3:end));
    
    % Plot Parameter Estimations
    figure;
    plot(t, theta_hat(:,3:end), 'LineWidth', 1.5);
    legend({'$\hat{\alpha}_{11}$', '$\hat{\alpha}_{12}$', ...
            '$\hat{\alpha}_{21}$', '$\hat{\alpha}_{22}$', ...
            '$\hat{b}_1$', '$\hat{b}_2$'}, 'Interpreter', 'latex');
    grid on;
    xlabel('t [sec]');
    title(sprintf('Parameter Estimations (Input: %s)', inputNames{i}));

    % Export files
    filename = fullfile(outputDir, sprintf('task1_parameter_estimations_%s.pdf', lower(inputNames{i})));
    exportgraphics(gcf, filename, 'ContentType', 'vector');
    
    %% Plot Identification Error
    n_states = size(x, 2);
    figure('Position', [200, 100, 800, 400]);
    sgtitle(sprintf('Identification Error (Input: %s)', inputNames{i}));
    for j = 1:n_states
        x_hat = theta_hat(:,1:2);
        e = x - x_hat;
        subplot(1, 2, j);
        plot(t, [x(:,j), x_hat(:,j), e(:,j)], 'LineWidth', 1.5);
        legend({sprintf('$x_%d$', j), ...
                sprintf('$\\hat{x}_%d$', j), ...
                sprintf('$e_%d$', j)}, 'Interpreter', 'latex');
        grid on;
        xlabel('t [sec]');
        title(sprintf('State %d', j));
    end

    % Export files
    filename = fullfile(outputDir, sprintf('task1_identification_error_%s.pdf', lower(inputNames{i})));
    exportgraphics(gcf, filename, 'ContentType', 'vector');

    %% Plot Lyapunov function along with its derivative
    figure;
    plot(V, 'LineWidth', 1.5);
    xlabel('t [sec]');
    title(sprintf('Lyapunov Function and Derivative (Input: %s)', inputNames{i}));
    legend({'$V$', '$\dot{V}$'}, 'Interpreter', 'latex');
    grid on;

    % Export files
    filename = fullfile(outputDir, sprintf('task1_lyapunov_%s.pdf', lower(inputNames{i})));
    exportgraphics(gcf, filename, 'ContentType', 'vector');
end

