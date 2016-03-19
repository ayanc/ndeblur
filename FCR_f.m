% Fully Connected layer + ReLU: forward
% --Ayan Chakrabarti <ayanc@ttic.edu>
function my_acts = FCR_f(params,wts,activations,inputs)

btms = params.btms;

act = 0;
for i = 1:length(btms)
  if btms(i) < 0
    act = act + wts{i}*inputs{-btms(i)};
  else
    act = act + wts{i} * activations{btms(i)}{1};
  end;
end;
if length(btms) < length(wts) 
  act = bsxfun(@plus,act,wts{end});
end;

my_acts = {max(0,act)};
