function phi = generate_regressor(x, u, type, params)
    % Create basis functions for input x, u
    % type: 'poly' or 'gauss'
    % params:
    %   for 'poly':   params.order: highest degree
    %   for 'gauss':  params.centers, params.width

    x = x(:); u = u(:);  % ensure column vectors
    N = length(x);

    switch lower(type)
        case 'poly'
            order = params.order;
            phi = zeros(N, order + 1);
            for i = 1:order
                phi(:, i) = x.^i;
            end
            phi(:, end) = u;

        case 'gauss'
            c = params.centers(:);    % column vector of centers
            w = params.width;         % scalar or vector
            num_g = length(c);
            phi = zeros(N, num_g + 1);
            for i = 1:num_g
                phi(:, i) = exp(-((x - c(i)).^2) / (2 * w^2));
            end
            phi(:, end) = u;

        otherwise
            error('Unknown basis type "%s". Use "poly" or "gauss".', type);
    end
end
