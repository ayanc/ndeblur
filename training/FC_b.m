% Fully Connected layer: backward
% --Ayan Chakrabarti <ayanc@ttic.edu>
function [grad,btm_loss] = FC_b(params,wts,activations,inputs,my_loss)

btms = params.btms;

btm_loss = cell(length(btms),1);
grad = cell(length(wts),1);

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


