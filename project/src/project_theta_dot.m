function projected_dot = project_theta_dot(theta_dot, theta)
% PROJECT_THETA_DOT Applies projection to the parameter derivative vector
% Inputs:
%   theta_dot - 6x1 vector: [a11_dot; a21_dot; a12_dot; a22_dot; b1_dot; b2_dot]
%   theta     - 6x1 vector: [a11;     a21;     a12;     a22;     b1;     b2    ]
%
% Output:
%   projected_dot - 6x1 vector with projection applied

    % Constraint functions:
    % g1 = -a11 - 3 <= 0
    % g2 =  a11 + 1 <= 0
    % g3 =  1 - b2 <= 0

    % Constraint gradients
    grad_g = [ ...
        -1, 0, 0, 0, 0, 0;
         1, 0, 0, 0, 0, 0;
         0, 0, 0, 0, 0, -1
    ];

    % Constraint values
    g = [-theta(1) - 3;
          theta(1) + 1;
          1 - theta(6)];

    Gamma = eye(6);  % Adaptation gain (identity for simplicity)

    projected_dot = theta_dot;

    for i = 1:length(g)
        gi = g(i);
        grad_i = grad_g(i,:)';

        % Project if constraint is active and pushing outward
        if gi >= 0 && grad_i' * theta_dot > 0
            correction = Gamma * (grad_i * grad_i') / (grad_i' * Gamma * grad_i) * theta_dot;
            projected_dot = projected_dot - correction;
        end
    end
end
