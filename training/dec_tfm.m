% tfm.inp = dec_tfm(tfm.inp,patches,kernels)
%   patches should be size A x A x N,
%   kernels should be size K x K x M
%   such that A-K+1 = 65
% --Ayan Chakrabarti <ayanc@ttic.edu>
function dec = dec_tfm(inputs,pdata,kernels)

NCHUNK = 100; CHUNK=1024;

dec = inputs;

cv = cell(length(inputs),1);
for i = 1:length(inputs)
  cv{i} = 0; inputs{i} = gpuArray(single(inputs{i}));
end;

for i = 1:NCHUNK
  
  pidx = mod([(i-1)*CHUNK+1:i*CHUNK]-1,size(pdata,3))+1;
  kidx = mod([(i-1)*CHUNK+1:i*CHUNK]-1,size(kernels,3))+1;
  
  obs = gmConv( pdata(:,:,pidx), ...
		kernels(:,:,kidx) )-0.5;
  for j = 1:length(cv)
    x = inputs{j}*obs;
    cv{j} = cv{j} + cov(x')/NCHUNK;
  end;
end;

for i = 1:length(inputs)
   [v,d] = eig(gather(cv{i}));
   dec{i} = diag(1./sqrt(diag(d))) * v' * dec{i};
end;
