% function K = make_kernels(psz,mxsz,nc,num)
%
% Make what looks like a motion kernel
%   K will be of size mxsz x mxsz x num
%
%   nc "Control points" for spline will be sampled from a psz x psz
%   grid.
%
% In paper, we used three sets of kernels for training
% k1 = make_kernels(24,41,6,33333); 
% k2 = make_kernels(16,41,6,33333);
% k3 = make_kernels(8,41,6,33333);
% kernels = cat(3,k1,k2,k3);  
%
% --Ayan Chakrabarti <ayanc@ttic.edu>
function K = make_kernels(psz,mxsz,nc,num)

K = zeros([mxsz mxsz num],'single');

imp = zeros(mxsz,mxsz);
imp((mxsz+1)/2,(mxsz+1)/2) = 1;

[xg,yg] = meshgrid([1:psz]);

for i = 1:num
  fprintf('\r Generating %06d of %06d    ',i,num);
  
  while 1
  x = randi(psz,[1 nc]); y = randi(psz,[1 nc]);
  
  x = spline(linspace(0,1,nc),x,linspace(0,1,nc*5000));
  x = round(max(1,min(psz,x)));
  
  y = spline(linspace(0,1,nc),y,linspace(0,1,nc*5000));
  y = round(max(1,min(psz,y)));
  
  idx = (x-1)*psz + y; idx = unique(idx);
  wt = max(0,randn(length(idx),1,1)*0.5+1);
  if sum(wt) == 0
    continue;
  end;
  
  wt = wt / sum(wt);
  
  krnl = zeros(psz,psz); krnl(idx) = wt;
  
  cx = round(sum(krnl(:).*xg(:))); 
  cy = round(sum(krnl(:).*yg(:)));
  
  if cx <= psz/2
    krnl = [zeros(psz,psz-2*cx+1) krnl];
  else
    krnl = [krnl zeros(psz,2*cx-psz-1)];
  end;
  p2 = size(krnl,2);
  
  if cy <= psz/2
    krnl = [zeros(psz-2*cy+1,p2); krnl];
  else
    krnl = [krnl; zeros(2*cy-psz-1,p2)];
  end;
  if max(size(krnl)) <= mxsz
    break;
  end;
  end;
  
  
  K(:,:,i) = single(conv2(imp,krnl,'same'));
end;
fprintf('\n');