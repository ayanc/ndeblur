% wts = deepcopy(wts,dir)
%
% dir = 0: Copy from cpu to gpu
%       1: Copy from gpu to cpu
% --Ayan Chakrabarti <ayanc@ttic.edu>
function wts = deepcopy(wts_i, dir)

wts = cell(length(wts_i),1);
for i = 1:length(wts_i)
  wts{i} = wts_i{i};
  for j = 1:length(wts{i})
    
    if dir == 0
      wts{i}{j} = gpuArray(single(wts{i}{j}));
    else
      wts{i}{j} = gather(wts{i}{j});
    end;
    
  end;
end;