% Updated weights by stochastic gradient descent
%   Called as a script instead of a function to avoid making copies
%   of weights and gradients.
%   Make sure the following variables are available in the caller:
%     net:    network cell array
%     wts:    current values of the weights
%     grad:   previous value of gradients
%     n_grad: value of gradients from last backward pass
%
%     lr:     Learning rate
%     mom:    Momentum
% --Ayan Chakrabarti <ayanc@ttic.edu>


for SGD_i = 1:length(net)

  % Gradient and weight update
  for SGD_j = 1:length(wts{SGD_i})
    grad{SGD_i}{SGD_j} = grad{SGD_i}{SGD_j}*mom	+ n_grad{SGD_i}{SGD_j};
    wts{SGD_i}{SGD_j} = wts{SGD_i}{SGD_j} - lr*grad{SGD_i}{SGD_j};
  end;

end;