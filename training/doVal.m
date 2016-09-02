%function verr = doVal(pdata, kernels,tfm, net,wts_in, N)
%
% --Ayan Chakrabarti <ayanc@ttic.edu>
function verr = doVal(pdata, kernels,tfm, net,wts_in, N)


% Copy tfm params to the gpu
for i = 1:length(tfm.inp)
  tfm.inp{i} = gpuArray(single(tfm.inp{i}));
end;
tfm.full = gpuArray(single(tfm.full));
tfm.ifull = tfm.full(:,tfm.lpos)';


% Copy weights to gpu
wts = deepcopy(wts_in,0);

%%%%%% Run iterations
inputs = {}; verr = 0; oerr = 0;
for i = 1:N
  % Convolve and generate data
  [obs,gt] = gmConv( pdata, kernels );

  % Original error
  oerr = oerr + mean(mean( (gt(tfm.lpos,:) - obs(tfm.lpos,:)).^2 ));

  % Set up output targets
  obFT = tfm.full*obs;

  gt = bsxfun(@minus,gt(tfm.lpos,:),mean(obs,1));
  % Set up inputs to network
  obs = obs - 0.5;
  for j = 1:length(tfm.inp)
    inputs{j} = tfm.inp{j}*obs;
  end;
  
  % Forward pass (TEST PHASE)
  activations = doForward(net,wts,inputs,true);
  % Use predicted filter to deconvolve
  pred = tfm.ifull * cMul(obFT,activations{end}{1});

  % Compute Loss
  loss = pred-gt;
  verr = verr + gather(mean(loss(:).^2));
  
  kernels = kernels(:,:,[end 1:end-1]);
end;

verr = verr / N; oerr = oerr / N;
fprintf(' Validation Error: %.4e (%.4e)\n',verr,oerr);
