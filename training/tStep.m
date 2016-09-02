%function tStep(pdata,kernels,vdata,vkernels,tfm,...
%	        wname,lr,mom,bsize,steps)
%
% pdata: Training patches: 105x105XN
% kernels: Training Kernels: 41x41xN2
% vdata: Validation patches: 105x105XM
% vkernels: Validation Kernels: 41x41xM (same no. of validation
%           kernels and patches) 
% tfm: Transform (after doing dec_tfm)
% wname: Basename of matlab file with initialized/stored weights.
% lr: Learning Rate
% mom: Momentum
% bsize: Batch Size
% steps: Number of 1000 iteration steps.
%
% --Ayan Chakrabarti <ayanc@ttic.edu>
function tStep(pdata,kernels,vdata,vkernels,tfm,...
	       wname,lr,mom,bsize,steps)

N = 1000; vN = 10;
verr = []; terr = []; lrs = []; bsizes = []; moms = [];
best_i = 0; vbest = Inf; best_wts = {};

load([wname '.mat']);

for k = 1:steps
  fprintf('**** Iteration 1000 x %d\n',length(terr)+1);
  lrs(end+1) = lr; moms(end+1) = mom; bsizes(end+1) = bsize;
  tic;
  [wts,grad,tmp] = doTrain(pdata,kernels,tfm,net,wts,grad,lr,mom,bsize,N);
  toc
  terr(end+1) = tmp;
  tmp = doVal(vdata,vkernels,tfm,net,wts,vN);
  verr(end+1) = tmp;
  
  if tmp < vbest
    vbest = tmp;
    best_wts = wts;
    best_i = length(verr);
  end;

end;

save('-mat',[wname '.mat'],'net','wts','grad','verr','terr','lrs','bsizes','moms','vbest','best_i','best_wts');
