clc, clearvars, close all;
addpath('util\');
outputDir = fullfile('..', 'plot');

% Define parameters
m = 0.75;
L = 1.25;
c = 0.15;
g = 9.81;
A0 = 4;
w = 2;

% Simulation start and end
tstart = 0;
tend = 20;

%% Task 1

[t, x] = simulateSystemResponse(m, L, c, g, A0, w, [tstart, tend]);
legendText = {'$q(t)$', '$\dot{q}(t)$'};
fileName = 'task1_system_response.pdf';
plotAndExport(t, x, 't (sec)', '', 'System Response', outputDir, fileName, legendText);
fileName = 'task1_system_states.pdf';
plotAndExport(x(:,1), x(:,2), '$q(t)$', '$\dot{q}(t)$', 'System States', outputDir, fileName);


%% Task 2

% Perform sampling on state vector
Ts = 0.1;                 % Sampling period
t_s = tstart:Ts:tend;     % Sampling times
u_s = A0 * sin(w * t_s);  % Input sample
x_s = interp1(t, x, t_s);

%% a) The output derivative is available
p1 = 0.2;
p2 = 0.6;
[c_hat1, L_hat1, m_hat1] = estimateParameters(u_s, x_s, t_s, true, p1, p2, g);

% Simulate system response using the estimated parameters
[~, x_hat1] = simulateSystemResponse(m_hat1, L_hat1, c_hat1, g, A0, w, t_s);
e1 = x_s(:,1) - x_hat1(:,1);
rmse1 = sqrt(mean(e1.^2));
fileName = 'task2_response_with_derivative.pdf';
legendText = {'$q_s$', '$\dot{q}_s$', '$\hat{q}$', '$\dot{\hat{q}}$'};
plotAndExport(t_s, [x_s, x_hat1], '$t_s$ (sec)', '', 'Estimated System Response (Output derivative available)', outputDir, fileName, legendText);

fprintf('\nTask 2\n');
fprintf('=============\n');
fprintf('a) Output derivative available:\n');
fprintf('c_hat: %f\n', c_hat1);
fprintf('L_hat: %f\n', L_hat1);
fprintf('m_hat: %f\n', m_hat1);
fprintf('RMSE: %f\n', rmse1);


%% b) The output derivative is NOT available
[c_hat2, L_hat2, m_hat2] = estimateParameters(u_s, x_s, t_s, false, p1, p2, g);

% Simulate system response using the estimated parameters
[~, x_hat2] = simulateSystemResponse(m_hat2, L_hat2, c_hat2, g, A0, w, t_s);
e2 = x_s(:,1) - x_hat2(:,1);
rmse2 = sqrt(mean(e2.^2));
fileName = 'task2_response_without_derivative.pdf';
legendText = {'$q_s$', '$\dot{q}_s$', '$\hat{q}$', '$\dot{\hat{q}}$'};
plotAndExport(t_s, [x_s, x_hat2], '$t_s$ (sec)', '', 'Estimated System Response (Output derivative not available)', outputDir, fileName, legendText);

fprintf('b) Output derivative NOT available:\n');
fprintf('c_hat: %f\n', c_hat2);
fprintf('L_hat: %f\n', L_hat2);
fprintf('m_hat: %f\n', m_hat2);
fprintf('RMSE: %f\n', rmse2);

% Plot simulated system residuals
fileName = 'task2_residuals.pdf';
legendText = {'with $\dot{q}(t)$', 'without $\dot{q}(t)$'};
plotAndExport(t_s, [e1, e2], '$t_s$ (sec)', '$q(t) - \hat{q}(t)$', 'Estimated System Residuals', outputDir, fileName, legendText);


%% Task 3

%% a) Add White Gaussian Noise (WGN) to sampling data
sigma = 0.01;
x_s_wgn = x_s + sigma * randn(size(x_s));
u_s_wgn = u_s + sigma * randn(size(u_s));

fileName = 'task3_sampling_data_with_noise.pdf';
legendText = {'$q(t)+\eta$', '$\dot{q}(t)+\eta$', '$q(t)$', '$\dot{q}(t)$'};
plotAndExport(t_s, [x_s_wgn, x_s], 't (sec)', '', 'Sampling Data with WGN', outputDir, fileName, legendText);

%% i) The output derivative is available

[c_hat1_wgn, L_hat1_wgn, m_hat1_wgn] = estimateParameters(u_s_wgn, x_s_wgn, t_s, true, p1, p2, g);

% Simulate system response using the estimated parameters
[~, x_hat1_wgn] = simulateSystemResponse(m_hat1_wgn, L_hat1_wgn, c_hat1_wgn, g, A0, w, t_s);
e1_wgn = x_s(:,1) - x_hat1_wgn(:,1);
rmse1_wgn = sqrt(mean(e1_wgn.^2));
fileName = 'task3_response_with_derivative_WGN.pdf';
legendText = {'$q_s$', '$\hat{q}$', '$\hat{q}$ WGN'};
plotAndExport(t_s, [x_s(:,1), x_hat1(:,1), x_hat1_wgn(:,1)], '$t_s$ (sec)', '', 'Estimated System Response with WGN (Output derivative available)', outputDir, fileName, legendText);

fprintf('\nTask 3\n');
fprintf('=============\n');
fprintf('a) Output derivative available:\n');
fprintf('c_hat: %f\n', c_hat1_wgn);
fprintf('L_hat: %f\n', L_hat1_wgn);
fprintf('m_hat: %f\n', m_hat1_wgn);
fprintf('RMSE: %f\n', rmse1_wgn);

% Plot simulated system residuals
fileName = 'task3_residuals_with_derivative_WGN.pdf';
legendText = {'without WGN', 'with WGN'};
plotAndExport(t_s, [e1, e1_wgn], '$t_s$ (sec)', '$q(t) - \hat{q}(t)$', 'Estimated System Residuals (Output derivative available)', outputDir, fileName, legendText);

%% ii) The output derivative is NOT available
[c_hat2_wgn, L_hat2_wgn, m_hat2_wgn] = estimateParameters(u_s_wgn, x_s_wgn, t_s, false, p1, p2, g);

% Simulate system response using the estimated parameters
[~, x_hat2_wgn] = simulateSystemResponse(m_hat2_wgn, L_hat2_wgn, c_hat2_wgn, g, A0, w, t_s);
e2_wgn = x_s(:,1) - x_hat2_wgn(:,1);
rmse2_wgn = sqrt(mean(e2_wgn.^2));
fileName = 'task3_response_without_derivative_WGN.pdf';
legendText = {'$q_s$', '$\hat{q}$', '$\hat{q}$ WGN'};
plotAndExport(t_s, [x_s(:,1), x_hat2(:,1), x_hat2_wgn(:,1)], '$t_s$ (sec)', '', 'Estimated System Response with WGN (Output derivative not available)', outputDir, fileName, legendText);

fprintf('b) Output derivative NOT available:\n');
fprintf('c_hat: %f\n', c_hat2_wgn);
fprintf('L_hat: %f\n', L_hat2_wgn);
fprintf('m_hat: %f\n', m_hat2_wgn);
fprintf('RMSE: %f\n', rmse2_wgn);

% Plot simulated system residuals
fileName = 'task3_residuals_without_derivative_WGN.pdf';
legendText = {'without WGN', 'with WGN'};
plotAndExport(t_s, [e2, e2_wgn], '$t_s$ (sec)', '$q(t) - \hat{q}(t)$', 'Estimated System Residuals (Output derivative not available)', outputDir, fileName, legendText);

%% b) Parameter estimation accuracy vs Sampling Period
    
