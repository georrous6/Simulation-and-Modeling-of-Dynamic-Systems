function phi = generate_regressor(x, u, type, params)
    % Create basis functions for input x, u
    % type: 'poly', 'gauss', or 'cos'
    % params:
    %   for 'poly':   params.order
    %   for 'gauss':  params.centers, params.width
    %   for 'cos':    params.freqs (vector of frequencies)

    x = x(:); u = u(:);  % ensure column vectors
    N = length(x);

    switch lower(type)
        case 'poly'
            order = params.order;
            phi = zeros(N, order + 2);  % +1 for intercept, +1 for u
            phi(:, 1) = 1;              % intercept term
            for i = 1:order
                phi(:, i+1) = x.^i;
            end
            phi(:, end) = u;

        case 'gauss'
            c = params.centers(:);    % centers
            w = params.width;         % width
            num_g = length(c);
            phi = zeros(N, num_g + 2);  % +1 for intercept, +1 for u
            phi(:, 1) = 1;              % intercept term
            for i = 1:num_g
                phi(:, i+1) = exp(-((x - c(i)).^2) / (2 * w^2));
            end
            phi(:, end) = u;

        case 'cos'
            freqs = params.freqs(:);  % frequency vector
            num_f = length(freqs);
            phi = zeros(N, num_f + 2);  % +1 for intercept, +1 for u
            phi(:, 1) = 1;              % intercept term
            for i = 1:num_f
                phi(:, i+1) = cos(freqs(i) * x);
            end
            phi(:, end) = u;

        otherwise
            error('Unknown basis type "%s". Use "poly", "gauss", or "cos".', type);
    end
end
