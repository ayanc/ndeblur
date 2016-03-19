% function out = doepll(y,K,nz_std)
%
% Carry out non-blind deconvolution using EPLL (Zoran &
% Weiss). Please download epll separately and make sure it is in
% your matlab path.
%
% Input and output will be of same size, and assumed to be floats
% between 0,1. 
% --Ayan Chakrabarti <ayanc@ttic.edu>
function out = doEPLL(y,K,mse)

load('GSModel_8x8_200_2M_noDC_zeromean.mat');

noiseSD=0.01;
if exist('mse')
  noiseSD = mse;
end;
patchSize = 8;

ks = floor( (size(K,1)-1)/2 );
eflt =  fspecial('gaussian',2*ks+1,ks);
y = padarray(y, [1 1]*ks, 'replicate', 'both');
for a=1:4
  y = edgetaper(y,eflt);
end

excludeList = [];
prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);
LogLFunc = [];

y = double(y); K = double(K);

true_i = y;

% deblur
tic
out = EPLLhalfQuadraticSplitDeblur(y,64/noiseSD^2,K,patchSize,50*[1 2 4 8 16 32 64],1,prior,true_i,LogLFunc);
toc

out = out(1+ks:end-ks,1+ks:end-ks);