function phi = generate_regressor(x, u, type, params)
%GENERATE_REGRESSOR Constructs a regression matrix based on input data
%
%   phi = GENERATE_REGRESSOR(x, u, type, params) returns the regression matrix
%   phi, where each row is a feature vector formed from the scalar state x and
%   input u using the specified basis type and parameters.
%
%   Inputs:
%       x      - Nx1 vector of state values
%       u      - Nx1 vector of input values
%       type   - string specifying the basis type:
%                'poly'  - polynomial basis
%                'gauss' - Gaussian basis
%                'cos'   - cosine basis
%       params - structure containing the required parameters:
%           For 'poly':
%               params.order   - polynomial order
%           For 'gauss':
%               params.centers - vector of Gaussian centers
%               params.width   - scalar Gaussian width
%           For 'cos':
%               params.freqs   - vector of cosine frequencies
%
%   Output:
%       phi - NxM regression matrix with intercept, basis terms, and input u

    x = x(:); u = u(:);  % ensure column vectors
    N = length(x);

    switch lower(type)
        case 'poly'
            % Polynomial basis: [1, x, x^2, ..., x^order, u]
            order = params.order;
            phi = zeros(N, order + 2);  % +1 for intercept, +1 for u
            phi(:, 1) = 1;              % intercept term
            for i = 1:order
                phi(:, i+1) = x.^i;     % Polynomial terms
            end
            phi(:, end) = u;            % Control input

        case 'gauss'
            % Gaussian basis: [1, exp(-((x-c)^2)/(2w^2)), ..., u]
            c = params.centers(:);    % centers
            w = params.width;         % common width
            num_g = length(c);
            phi = zeros(N, num_g + 2);  % +1 for intercept, +1 for u
            phi(:, 1) = 1;              % intercept term
            for i = 1:num_g
                phi(:, i+1) = exp(-((x - c(i)).^2) / (2 * w^2));  % Gaussian term
            end
            phi(:, end) = u;  % Control input

        case 'cos'
            % Cosine basis: [1, cos(w1*x), ..., u]
            freqs = params.freqs(:);    % Frequencies for cosine basis
            num_f = length(freqs);
            phi = zeros(N, num_f + 2);  % +1 for intercept, +1 for u
            phi(:, 1) = 1;              % intercept term
            for i = 1:num_f
                phi(:, i+1) = cos(freqs(i) * x);  % cosine terms
            end
            phi(:, end) = u;  % Control input

        otherwise
            error('Unknown basis type "%s". Use "poly", "gauss", or "cos".', type);
    end
end
