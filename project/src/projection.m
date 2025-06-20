function theta_hat_dot_proj = projection(theta_hat_dot, theta_hat, Gamma)
% PROJECT Applies projection to the parameter derivative vector
% Inputs:
%   theta_dot - 6x1 vector: [a11_dot; a12_dot; a21_dot; a22_dot; b1_dot; b2_dot]
%   theta     - 6x1 vector: [a11;     a12;     a21;     a22;     b1;     b2    ]
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
    g = [-theta_hat(1) - 3;
          theta_hat(1) + 1;
          1 - theta_hat(6)];

    theta_hat_dot_proj = theta_hat_dot;

    for i = 1:length(g)
        gi = g(i);
        grad_i = grad_g(i,:)';

        % Project if constraint is active and pushing outward
        if gi >= 0 && grad_i' * theta_hat_dot > 0
            correction = Gamma * (grad_i * grad_i') / (grad_i' * Gamma * grad_i) * theta_hat_dot;
            theta_hat_dot_proj = theta_hat_dot_proj - correction;
        end
    end
end
