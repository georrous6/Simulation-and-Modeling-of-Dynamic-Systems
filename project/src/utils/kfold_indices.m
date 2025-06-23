function indices = kfold_indices(n, K)
    % Manually assign data indices to K folds
    indices = zeros(n,1);
    idx = randperm(n);         % random shuffle
    fold_sizes = floor(n / K) * ones(1, K);
    fold_sizes(1:mod(n,K)) = fold_sizes(1:mod(n,K)) + 1;  % handle leftovers

    start_idx = 1;
    for k = 1:K
        end_idx = start_idx + fold_sizes(k) - 1;
        indices(idx(start_idx:end_idx)) = k;
        start_idx = end_idx + 1;
    end
end
