% Fully Connected layer + ReLU: backward
% --Ayan Chakrabarti <ayanc@ttic.edu>
function [grad,btm_loss] = FCR_b(params,wts,activations,inputs,my_loss)

btms = params.btms;
my_idx = params.my_idx;

btm_loss = cell(length(btms),1);
grad = cell(length(wts),1);

my_loss = my_loss .* (activations{my_idx}{1} > 0);

for i = 1:length(btms)
  if btms(i) < 0
    grad{i} = my_loss*inputs{-btms(i)}';
  else
    grad{i} = my_loss*activations{btms(i)}{1}';
    btm_loss{i} = wts{i}'*my_loss;
  end;
end;

if length(btms) < length(wts) 
  grad{end} = sum(my_loss,2);
end;


