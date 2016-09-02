%function [wts,grad,terr] = doTrain(pdata, kernels, ...
% 				    tfm, net, ...
%				    wts_in, grad_in, ...
%				    lr, mom, bsize, N)
%
% --Ayan Chakrabarti <ayanc@ttic.edu>
function [wts,grad,terr] = doTrain(pdata, kernels, ...
				   tfm, net, ...
				   wts_in, grad_in, ...
				   lr, mom, bsize, N)


% Randomly select images and kernels for training
kidx = uint32(randi(size(kernels,3),[N bsize]));
pidx = uint32(randi(size(pdata,3),[N bsize]));

% Copy tfm params to the gpu
for i = 1:length(tfm.inp)
  tfm.inp{i} = gpuArray(single(tfm.inp{i}));
end;
tfm.full = gpuArray(single(tfm.full));
tfm.ifull = tfm.full(:,tfm.lpos)';


% Copy weights and gradients to gpu
wts = deepcopy(wts_in,0);
grad = deepcopy(grad_in,0);


%%%%%% Run iterations
inputs = {}; terr = 0; count = 0;
lnorm = length(tfm.lpos)*bsize;
for i = 1:N
  
  % Setup input and target
  pdi = pdata(:,:,pidx(i,:));

  % Do random transformations
  tfid = mod((i-1),8);
  if tfid > 3
    tfid = tfid - 4;
    pdi = permute(pdi,[2 1 3]);
  end;
  if tfid == 1 || tfid == 3
    pdi = pdi(end:-1:1,:,:);
  end;
  if tfid == 2 || tfid == 3
    pdi = pdi(:,end:-1:1,:);
  end;
  
  % Convolve and generate data
  [obs,gt] = gmConv( pdi, kernels(:,:,kidx(i,:)) );

  % Set up output targets
  obFT = tfm.full*obs;
  gt = bsxfun(@minus,gt(tfm.lpos,:),mean(obs,1));
  
  % Set up inputs to network
  obs = obs - 0.5;
  for j = 1:length(tfm.inp)
    inputs{j} = tfm.inp{j}*obs;
  end;
  
  % Forward pass
  activations = doForward(net,wts,inputs);
  
  % Use predicted filter to deconvolve
  pred = tfm.ifull * cMul(obFT,activations{end}{1});

  % Compute Loss
  loss = max(-1,min(1,pred-gt));
  terr = terr + gather(mean(loss(:).^2)); count = count + 1;
  fprintf('\r Iteration %05d, Loss = %.4e ',i,terr/count);
  
  % Backpropagate
  loss = tfm.ifull' * (0.5*loss/lnorm);
  loss = cMulC(loss,obFT);
  
  n_grad = doBackward(net,wts,activations,inputs, ...
		      { {length(activations), loss} });
  
  SGD;
  n_grad = {};
end;

terr = terr / count;
wts = deepcopy(wts,1);
grad = deepcopy(grad,1);

fprintf('\n');
