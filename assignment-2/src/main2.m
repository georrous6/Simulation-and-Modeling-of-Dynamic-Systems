clc, clearvars, close all;

addpath('utils\');
outputDir = fullfile('..', 'plot');

% Check if the output directory exists, if not, create it
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

a1 = 1.315;
a2 = 0.725;
a3 = 0.225;
b  = 1.175;

%% Feedback controller design

phi_0 = 0.4;
phi_inf = 0.05;
lambda = 0.4;
rho = 0.5;
k1 = 1.2;
k2 = 2.0;

T_0 = 20;

T = @(z) log((1 + z) / (1 - z));
phi = @(t) (phi_0 - phi_inf) * exp(-lambda * t) + phi_inf;
rd = @(t)(-pi / 1000) * t.^2 + (pi / 50) * t;  % desired trajectory
z1 = @(t, x) (x(1) - rd(t)) / phi(t);
alpha = @(t, x) -k1 * T(z1(t, x));
z2 = @(t, x) (x(2) - alpha(t, x)) / rho;
u = @(t, x) -k2 * T(z2(t, x));

x_0 = [0; 0];
odefun = @(t, x) [x(2); -a1 * x(2) - a2 * sin(x(1)) + a3 * x(2)^2 * sin(2 * x(1)) + b * u(t, x)];
[t, x] = ode45(odefun, [0, T_0], x_0);
r = x(:,1);
r_dot = x(:,2);

% Plot
figure;
grid on; hold on;
plot(t, r, 'b');
plot(t, rd(t), '--r');
xlabel('Time [s]'); ylabel('Roll angle [rad]');
legend('r(t)', 'r_d(t)');
title('Roll Angle Response');
filePath = fullfile(outputDir, 'task2_roll_angle_response.pdf');
exportgraphics(gcf, filePath, 'ContentType', 'vector');

figure;
grid on; hold on;
plot(t, abs(r - rd(t)), 'b');
plot(t, phi(t), '--r');
xlabel('t');
legend({'$|r(t) - r_d(t)|$', '$\phi(t)$'}, 'Interpreter', 'latex');
title('Position Tracking Accuracy');
filePath = fullfile(outputDir, 'task2_position_tracking_accuracy.pdf');
exportgraphics(gcf, filePath, 'ContentType', 'vector');

N = length(t);
alpha_values = NaN(N, 1);
for i = 1:N
    alpha_values(i) = alpha(t(i), x(i,:)');
end

figure;
grid on; hold on;
plot(t, abs(r_dot - alpha_values), 'b');
plot(t, rho * ones(size(t)), '--r');
xlabel('t');
legend({'$|\dot{r}(t) - \alpha(t)|$', '$\rho$'}, 'Interpreter', 'latex');
title('Velocity Tracking Accuracy');
filePath = fullfile(outputDir, 'task2_velocity_tracking_accuracy.pdf');
exportgraphics(gcf, filePath, 'ContentType', 'vector');

M = 5;
phi_inf_values = linspace(0.1 * phi_0, 0.9 * phi_0, M);
r_error = NaN(N, M);
labels = cell(1, M);
for i = 1:M

    phi_inf = phi_inf_values(i);
    labels{i} = ['$\phi_{\infty} = $' num2str(phi_inf)];

    T = @(z) log((1 + z) / (1 - z));
    phi = @(t) (phi_0 - phi_inf) * exp(-lambda * t) + phi_inf;
    rd = @(t)(-pi / 1000) * t.^2 + (pi / 50) * t;  % desired trajectory
    z1 = @(t, x) (x(1) - rd(t)) / phi(t);
    alpha = @(t, x) -k1 * T(z1(t, x));
    z2 = @(t, x) (x(2) - alpha(t, x)) / rho;
    u = @(t, x) -k2 * T(z2(t, x));
    odefun = @(t, x) [x(2); -a1 * x(2) - a2 * sin(x(1)) + a3 * x(2)^2 * sin(2 * x(1)) + b * u(t, x)];

    [~, x] = ode45(odefun, t, x_0);
    r_error(:,i) = abs(x(:,1) - rd(t));
end

figure;
hold on; grid on;
plot(t, r_error);
xlabel('t');
ylabel('$|r(t) - r_d(t)|$', 'Interpreter', 'latex');
legend(labels, 'Interpreter', 'latex');
title('Position Tracking Accuracy vs $\phi_{\infty}$', 'Interpreter', 'latex');
filePath = fullfile(outputDir, 'task2_position_tracking_accuracy_vs_phi_infty.pdf');
exportgraphics(gcf, filePath, 'ContentType', 'vector');
