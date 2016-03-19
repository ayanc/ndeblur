%function imout = pforward(img,tfm,net,wts_in)
% Apply neural network on all patches in input, and average their
% outputs.
% --Ayan Chakrabarti <ayanc@ttic.edu>
function imout = pforward(img,tfm,net,wts_in)

CHUNK=1024;
PSZ = 65;
CSZ=33;

% Center-window stuff
[yshft,xshft] = ndgrid([-(PSZ-1)/2:(PSZ-1)/2],[-(PSZ-1)/2:(PSZ-1)/2]);
yshft = yshft(:); xshft = xshft(:);
ridx = find(abs(yshft) <= (CSZ-1)/2 & abs(xshft) <= (CSZ-1)/2);
yshft = yshft(ridx); xshft = xshft(ridx);


% Copy tfm params to the gpu
for i = 1:length(tfm.inp)
  tfm.inp{i} = gpuArray(single(tfm.inp{i}));
end;
tfm.full = gpuArray(single(tfm.full));
tfm.ifull = tfm.full(:,ridx)';


% Copy weights to gpu
wts = deepcopy(wts_in,0);


% Set up chunking
sz = size(img); osz = sz - PSZ+1;
[y,x] = ndgrid([1:osz(1)],[1:osz(2)]);

oidx = uint32(y(:)' + (x(:)'-1)*osz(1));
iidx = uint32(y(:)' + (x(:)'-1)*sz(1));


[y,x] = ndgrid([0:PSZ-1],[0:PSZ-1]);
pidx = uint32(y(:) + x(:)*sz(1));

imout = zeros([osz(1)*osz(2) length(ridx)],'single','gpuArray');

%%%%%% Run iterations
inputs = {}; verr = 0; oerr = 0;
for i = 1:CHUNK:length(oidx)
  
  widx = [i:min(length(oidx),(i+CHUNK-1))];
  cpidx = bsxfun(@plus,iidx(widx),pidx);
  obs = gpuArray(single(img(cpidx)));
  
  obFT = tfm.full*obs;
  obMean = mean(obs,1);
  
  % Set up inputs to network
  obs = obs - 0.5;
  for j = 1:length(tfm.inp)
    inputs{j} = tfm.inp{j}*obs;
  end;
  
  % Forward pass (TEST PHASE)
  activations = doForward(net,wts,inputs,true);
  % Use predicted filter to deconvolve
  pred = cMul(obFT,activations{end}{1});
  pred = bsxfun(@plus,tfm.ifull * pred, obMean);
  
  imout(oidx(widx),:) = pred';
  fprintf('\r %10d of %10d pixels   ',widx(end),length(oidx));
  
end;
clear pred activations inputs obFT obmean

imout = reshape(imout,[osz length(ridx)]);
imz = zeros([osz+(CSZ-1) length(ridx)],'single','gpuArray');
for i = 1:length(yshft)

  ylim = [1-yshft(i)-(CSZ-1)/2:osz(1)-yshft(i)+(CSZ-1)/2];
  xlim = [1-xshft(i)-(CSZ-1)/2:osz(2)-xshft(i)+(CSZ-1)/2];
  
  ylim = max(1,min(osz(1),ylim));
  xlim = max(1,min(osz(2),xlim));
  
  imz(:,:,i) = imout(ylim,xlim,i);
end;
clear imout

% Average with hanning window
gm = hanning(CSZ); gm = gm*gm'; gm = gm / sum(gm(:));
gm = reshape(gm,[1 1 CSZ^2]);
imz = max(0,min(1,imz));
imout = sum(bsxfun(@times,imz,gm),3);

imout = gather(imout);
fprintf('\n');