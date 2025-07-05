function u = stochastic_switch(t_span, T_dwell)
%STOCHASTIC_SWITCH Generate piecewise-constant binary input signal
%
%   u = STOCHASTIC_SWITCH(t_span, T_dwell) returns a function handle u(tau)
%   that provides a binary signal (0 or 1) that switches randomly every
%   T_dwell seconds over the interval specified by t_span.
%
%   Inputs:
%       t_span   - 1x2 vector specifying start and end times [t0, tf]
%       T_dwell  - dwell time between switches (positive scalar)
%
%   Output:
%       u        - function handle u(tau) giving binary signal value at time tau

    t0 = t_span(1);
    tf = t_span(end);
    t_switches = t0:T_dwell:tf;
    values = randi([0, 1], size(t_switches));
    u = @(tau) interp1(t_switches, values, tau, 'previous', 'extrap');
end