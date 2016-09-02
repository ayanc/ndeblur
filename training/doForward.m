% Do forward pass over network and compute all activations
% --Ayan Chakrabarti <ayanc@ttic.edu>
function activations = doForward(net,wts,inputs,iftest)

if exist('iftest')
  for i = 1:length(net)
    net{i}{2} = setfield(net{i}{2},'test',1);
  end;
end;

activations = cell(length(net),1);
for i = 1:length(activations)
  activations{i}=net{i}{2}.fwd(net{i}{2},wts{i},activations,inputs);
end;