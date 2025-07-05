function sigma = sigma_switch(theta_bar, M, sigma_bar)
%SIGMA_SWITCH Compute sigma-modification gain based on parameter norm
%
%   sigma = SIGMA_SWITCH(theta_bar, M, sigma_bar) returns a gain value used
%   in adaptive parameter update laws to prevent parameter drift. The gain 
%   increases smoothly once the norm of theta_bar exceeds the threshold M,
%   and saturates at sigma_bar when norm(theta_bar) > 2*M.
%
%   Inputs:
%       theta_bar - parameter vector
%       M         - norm threshold for activation
%       sigma_bar - maximum sigma value
%
%   Output:
%       sigma     - sigma-modification gain

    norm_theta_bar = norm(theta_bar);
    if norm_theta_bar < M
        sigma = 0;
    elseif norm_theta_bar <= 2*M
        sigma = sigma_bar * (norm_theta_bar / M - 1);
    else
        sigma = sigma_bar;
    end
end