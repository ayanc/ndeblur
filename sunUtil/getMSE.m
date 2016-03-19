%function [mse,best_crop] = getMSE(est,ref)
% Find MSE between best crop of est and ground truth reference
% image ref. This best crop is also returned.
% --Ayan Chakrabarti <ayanc@ttic.edu>
function [mse,crop] = getMSE(est,ref)

est = double(est); ref = double(ref);
tmp = sum(ref(:).^2) - 2*conv2(est,fliplr(flipud(ref)),'valid') + ...
      conv2(est.^2,ones(size(ref)),'valid');

[mse,idx] = min(tmp(:));

[yl,xl] = ind2sub(size(tmp),idx);

yl = [1:size(ref,1)] - 1 + yl;
xl = [1:size(ref,2)] - 1 + xl;
crop = est(yl,xl);
lc = {yl, xl};

mse = crop-ref; mse = mean(mse(:).^2);
crop = single(crop);
