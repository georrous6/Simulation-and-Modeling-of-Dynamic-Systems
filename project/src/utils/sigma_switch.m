function sigma = sigma_switch(theta_bar, M, sigma_bar)
    norm_theta_bar = norm(theta_bar);
    if norm_theta_bar < M
        sigma = 0;
    elseif norm_theta_bar <= 2*M
        sigma = sigma_bar * (norm_theta_bar / M - 1);
    else
        sigma = sigma_bar;
    end
end