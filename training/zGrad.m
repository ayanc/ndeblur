% grad = zGrad(wts)
%
% Create zero-filled gradient blobs with same shape as wts
% --Ayan Chakrabarti <ayanc@ttic.edu>
function grad = zGrad(wts)

grad = cell(length(wts),1);
for i = 1:length(wts)
  grad{i} = cell(1,length(wts{i}));
  for j = 1:length(grad{i})
    grad{i}{j} = zeros(size(wts{i}{j}),'single');
  end;
end;