clc, clearvars, close all;

addpath('utils\');

m = 1.315;
b = 0.225;
k = 0.725;

m_0 = 1.0;
b_0 = 0.17;
k_0 = 0.5;

T = 20;
N = 100;

t = linspace(0, T, N)';

u1 = @(t) 2.5 * ones(size(t));
u2 = @(t) 2.5 * sin(t);

odefun1 = @(t, x) ([0, 1; -k/m, -b/m] * x(:) + [0; 1/m] * u1(t));
odefun2 = @(t, x) ([0, 1; -k/m, -b/m] * x(:) + [0; 1/m] * u2(t));

y0 = [0; 0];
[~, y1] = ode45(odefun1, t, y0);
[~, y2] = ode45(odefun2, t, y0);

p1 = -0.1; % -0.1
p2 = -0.2; % -0.2
Lambda = [1, -(p1 + p2), p1 * p2];
gamma = 1e-4;

%% Input 1
[m_hat1, b_hat1, k_hat1, y_hat1] = estimateParametersGradientDescend(y1, u1(t), t, m_0, b_0, k_0, Lambda, gamma);

figure(1);
hold on; grid on;
plot(t, [m_hat1, b_hat1, k_hat1], 'LineWidth', 1);
plot(t, repmat([m, b, k], N, 1), '--r', 'LineWidth', 1);
legend({'$\hat{m}$', '$\hat{b}$', '$\hat{k}$'}, 'Interpreter', 'latex');
xlabel('t');
title('Parameter estimations over time for input signal: u(t)=2.5');

e1 = y1(:,1) - y_hat1;
figure(2);
hold on; grid on;
plot(t, [y1(:,1), y_hat1, e1], 'LineWidth', 1);
legend({'$y$', '$\hat{y}$', 'e'}, 'Interpreter', 'latex');
xlabel('t');
title('Simulation error over time for input signal: u(t)=2.5');

% %% Input 2
% [m_hat2, b_hat2, k_hat2, y_hat2] = estimateParametersGradientDescend(y1, u1(t), t, m_0, b_0, k_0, Lambda, gamma);
% 
% figure(3);
% hold on; grid on;
% plot(t, [m_hat2, b_hat2, k_hat2], 'LineWidth', 1);
% plot(t, repmat([m, b, k], N, 1), '--r', 'LineWidth', 1);
% legend({'$\hat{m}$', '$\hat{b}$', '$\hat{k}$'}, 'Interpreter', 'latex');
% xlabel('t');
% title('Parameter estimations over time for input signal: u(t)=2.5*sin(t)');
% 
% e2 = y2(:,1) - y_hat2;
% figure(4);
% hold on; grid on;
% plot(t, [y2(:,1), y_hat2, e2], 'LineWidth', 1);
% legend({'$y$', '$\hat{y}$', 'e'}, 'Interpreter', 'latex');
% xlabel('t');
% title('Simulation error over time for input signal: u(t)=2.5*sin(t)');
