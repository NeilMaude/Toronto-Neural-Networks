function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    
    visible_data = sample_bernoulli(visible_data);

    % h_p size <number of hidden units> by <number of configurations that we're handling in parallel>.
    h_p = visible_state_to_hidden_probabilities(rbm_w, visible_data); 
    
    h = sample_bernoulli(h_p);

    d_1 = configuration_goodness_gradient(visible_data, h);

    % reconstruction
    v_p = hidden_state_to_visible_probabilities(rbm_w, h); 
    v = sample_bernoulli(v_p);

    h_p = visible_state_to_hidden_probabilities(rbm_w, v); 
    
    % Instead of a sampled state, we'll simply use the conditional probabilities.
    h = sample_bernoulli(h_p); 

    d_2 = configuration_goodness_gradient(v, h_p); 
    % use h_p instead of h, as per improvement in question 8
    
    ret = d_1 - d_2;
    
end
