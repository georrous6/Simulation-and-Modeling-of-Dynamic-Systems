function [x_train, u_train, x_test, u_test] = train_test_split(x, u, train_frac)
    assert(train_frac > 0 && train_frac < 1, ...
        sprintf('Training fraction should be between (0, 1): %f', train_frac));

    n = length(u);
    n_train = floor(train_frac * n);
    x_train = x(1:n_train);
    u_train = u(1:n_train);
    x_test = x(n_train+1:end);
    u_test = u(n_train+1:end);
end