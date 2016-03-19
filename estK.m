%function [k,mse] = estK(imb,ims,ksz)
%
% Estimate kernel given blurry input and estimate of sharp image
% ims, and max kernel size ksz (51 for Sun et al.).
% --Ayan Chakrabarti <ayanc@ttic.edu>
function [k,mse] = estK(imb,ims,ksz)

%%% Store for mse comp later
imb0 = single(imb);
ims0 = single(ims);

% Parameters
msz = floor(ksz/2);
etflt = fspecial('gaussian',msz,msz/6);


% Crop center of blurry image
off = size(imb)-size(ims); off = off / 2;
imb = imb(1+off:end-off,1+off:end-off);

[y,x,grads] = buildXY(imb,ims,etflt);

% Set up k extraction idxs (cheaper than repeated psf2otf)
kyidx = mod([-(ksz-1)/2:(ksz-1)/2],size(y,1))+1;
kxidx = mod([-(ksz-1)/2:(ksz-1)/2],size(y,2))+1;

k = []; mse = Inf;

for reg = 2.^[6:0.25:8]
  [kr,mser] = doKOPT(kxidx,kyidx,x,y,imb0,ims0,reg,grads);

  if mser < mse
    fprintf('%.2f %.2e\n',reg,mser);
    mse = mser; k = kr;
  end;
end;


ims0 = conv2(ims0,k,'valid'); dsz = size(imb0,1)-size(ims0,1);
dsz = dsz/2;
imb0 = imb0(1+dsz:end-dsz,1+dsz:end-dsz);
mse = sqrt(mean((ims0(:)-imb0(:)).^2));



function [y,x,grads] = buildXY(imb,ims,etflt)

grads = {};
ors = linspace(0,180,9); ors = ors(1:end-1);

% First order
fltD = [-5:5].*exp(-([-5:5].^2)/2); fltD = fltD / sqrt(sum(fltD(:).^2));
fltS = exp(-([-5:5].^2)/2); fltS = fltS / sqrt(sum(fltS(:).^2));

imb_h = conv2(fltD,fltS,imb,'valid');
ims_h = conv2(fltD,fltS,ims,'valid');

imb_v = conv2(fltS,fltD,imb,'valid');
ims_v = conv2(fltS,fltD,ims,'valid');

idx = 1; msk = 0;
for d = ors
  grads{idx,1} = imb_h*cosd(d)+imb_v*sind(d);
  grads{idx,2} = ims_h*cosd(d)+ims_v*sind(d);
  msk = msk + grads{idx,2}.^2;
  
  idx = idx+1;
end;

% Second order
fltD = ([-5:5].^2-1).*exp(-([-5:5].^2)/2); 
nrm1 = sqrt(sum(fltD(:).^2));
fltS = exp(-([-5:5].^2)/2); 
nrm2 = sqrt(sum(fltS(:).^2));

xy = bsxfun(@times,[-5:5],[-5:5]');
xpy = bsxfun(@plus,[-5:5].^2,[-5:5]'.^2);
fltP = xy.*exp(-xpy/2) / nrm1 / nrm2;

fltD = fltD / nrm1; fltS = fltS /nrm2;

imb_h = conv2(fltD,fltS,imb,'valid');
ims_h = conv2(fltD,fltS,ims,'valid');

imb_v = conv2(fltS,fltD,imb,'valid');
ims_v = conv2(fltS,fltD,ims,'valid');

imb_p = conv2(imb,fltP,'valid');
ims_p = conv2(ims,fltP,'valid');

for d = ors
  grads{idx,1} = imb_h*cosd(d)^2+imb_v*sind(d)^2+2*imb_p*cosd(d)*sind(d);
  grads{idx,2} = ims_h*cosd(d)^2+ims_v*sind(d)^2+2*ims_p*cosd(d)*sind(d);
  msk = msk + grads{idx,2}.^2;
  idx = idx+1;
end;

thr = sort(msk(:)); thr = thr(round(0.98*length(thr)));
msk = msk > thr; 

y = 0; x = 0;
for i = 1:idx-1
  [y0,x0,g1,g2] = buildXY0(grads{i,1},grads{i,2}.*msk,etflt);
  y = y + y0; x = x + x0;
  grads{i,1} = g1; grads{i,2} = g2;
end;


function [y,x,g1,g2] = buildXY0(imb,ims,etflt)

% Take fft after tapering edges to prevent ringing
for a = 1:4
  imb = edgetaper(imb,etflt);
  ims = edgetaper(ims,etflt);
end;
imb = gpuArray(single(imb));
ims = gpuArray(single(ims));
y = fft2(imb); x = fft2(ims);
g1 = y; g2 = x;
y = y .* conj(x); x = x.*conj(x);


function [k,mse] = doKOPT(kxidx,kyidx,x,y,imb0,ims0,reg,grads)

kf = y ./ (x+eps);
kf = real(ifft2(kf));
k = kf(kyidx,kxidx); 

k0 = zeros(size(y),'single','gpuArray'); 
for beta = 2.^[-4:0.0625:32]
  k0(kyidx,kxidx) = k; kf = fft2(k0);
  
  kf = (y + beta*kf) ./ (x + beta);
  kf = real(ifft2(kf));
  k = kf(kyidx,kxidx);
  
  % Sparsify & project k
  k = k.* max(0,abs(k)-reg/beta) ./ max(eps,abs(k));
end;

k = gather(k); k = cleanK(k);

k0(kyidx,kxidx) = gpuArray(single(k)); kf = fft2(k0);
mse = 0;
for i = 1:size(grads,1)
  er = kf.*grads{i,2}-grads{i,1};
  mse = mse + gather(sum(er(:).*conj(er(:))));
end;


function k = cleanK(k)

thr = max(k(:))*1e-3;
k(k < thr) = 0; k = k /sum(k(:)); 

CC = bwconncomp(k,8);
for ii=1:CC.NumObjects
  csum=sum(k(CC.PixelIdxList{ii}));
  if csum < 0.1
    k(CC.PixelIdxList{ii}) = 0;
  end
end
k = k /sum(k(:));
