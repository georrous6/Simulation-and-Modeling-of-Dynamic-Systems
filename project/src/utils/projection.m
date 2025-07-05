function theta_hat_dot_proj = projection(theta_hat_dot, theta_hat, Gamma)
%PROJECTION Apply projection operator to parameter derivative vector to enforce constraints
%
%   theta_hat_dot_proj = PROJECTION(theta_hat_dot, theta_hat, Gamma) modifies the input
%   parameter derivative vector theta_hat_dot by projecting it onto the feasible set
%   defined by inequality constraints on theta_hat, using the metric Gamma.
%
%   Inputs:
%       theta_hat_dot - 6x1 vector of parameter derivatives
%                       [a11_dot; a12_dot; a21_dot; a22_dot; b1_dot; b2_dot]
%       theta_hat     - 6x1 current parameter vector
%                       [a11; a12; a21; a22; b1; b2]
%       Gamma         - 6x6 positive definite weighting matrix for projection
%
%   Output:
%       theta_hat_dot_proj - 6x1 projected parameter derivative vector
%
%   Constraints enforced:
%       g1(theta_hat) = -a11 - 3 <= 0
%       g2(theta_hat) =  a11 + 1 <= 0
%       g3(theta_hat) =  1 - b2  <= 0

    % Define constraint gradients (each row corresponds to a constraint)
    grad_g = [ ...
        -1, 0, 0, 0, 0, 0;
         1, 0, 0, 0, 0, 0;
         0, 0, 0, 0, 0, -1
    ];

    % Evaluate constraints
    g = [-theta_hat(1) - 3;
          theta_hat(1) + 1;
          1 - theta_hat(6)];

    % Initialize projected derivative as original derivative
    theta_hat_dot_proj = theta_hat_dot;

    % Loop over each constraint
    for i = 1:length(g)
        gi = g(i);
        grad_i = grad_g(i,:)';

        % Check if constraint is active (gi >= 0) and derivative pushes outside feasible set
        if gi >= 0 && grad_i' * theta_hat_dot > 0
            % Projection correction using Gamma metric
            correction = Gamma * (grad_i * grad_i') / (grad_i' * Gamma * grad_i) * theta_hat_dot;
            theta_hat_dot_proj = theta_hat_dot_proj - correction;
        end
    end
end
