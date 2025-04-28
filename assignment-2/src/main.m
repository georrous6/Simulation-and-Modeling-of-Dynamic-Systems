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
y0 = [0; 0];

T0 = 10;
alpha0 = 0.001;
alpha1 = 140;

phi = NaN(N, 3);

for i = 1:length(inputs)

    u = inputs{i};
    odefun = @(t, x) ([0, 1; -k/m, -b/m] * x(:) + [0; 1/m] * u(t));
    [~, x] = ode45(odefun, t, y0);
    phi(:,1) = lsim(tf(1, Lambda), x(:,1), t(:));
    phi(:,2) = lsim(tf(1, Lambda), x(:,2), t(:));
    phi(:,3) = lsim(tf(1, Lambda), u(t), t(:));

    gamma = gammas(i);
    [m_hat, b_hat, k_hat, y_hat] = estimateParametersGradientDescend(x(:,1), phi, m_0, b_0, k_0, Lambda, gamma);
    
    figure;
    hold on; grid on;
    plot(t, [m_hat, b_hat, k_hat], 'LineWidth', 1);
    plot(t, repmat([m, b, k], N, 1), '--r', 'LineWidth', 1);
    legend({'$\hat{m}$', '$\hat{b}$', '$\hat{k}$'}, 'Interpreter', 'latex');
    xlabel('t');
    title(sprintf('Parameter estimations for input signal u(t)=%s (gamma=%f)', labels{i}, gamma));
    filePath = fullfile(outputDir, sprintf('parameter_estimations_%d.pdf', i));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');
    
    e = x(:,1) - y_hat;
    figure;
    hold on; grid on;
    plot(t, [x(:,1), y_hat, e], 'LineWidth', 1);
    legend({'$x$', '$\hat{x}$', 'e'}, 'Interpreter', 'latex');
    xlabel('t');
    title(sprintf('Identification error for input signal u(t)=%s (gamma=%f)', labels{i}, gamma));
    filePath = fullfile(outputDir, sprintf('identification_error_%d.pdf', i));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');
    
    figure;
    is_PE = persistenceOfExcitationCondition(phi, t, T0, alpha0, alpha1);
    filePath = fullfile(outputDir, sprintf('PE_eigenvalues_%d.pdf', i));
    exportgraphics(gcf, filePath, 'ContentType', 'vector');
    if is_PE
        fprintf('Persistence of Excitation Condition is satisfied for u(t)=%s\n', labels{i});
    else
        fprintf('Persistence of Excitation Condition is NOT satisfied for u(t)=%s\n', labels{i});
    end
end