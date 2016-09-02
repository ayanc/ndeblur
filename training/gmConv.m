% [y,x2] = gmConv(x,k)
%  Create y(:,i) = x(:,i) * k(:,i) + noise
%  Valid convolution
%  x2 = x(:,i) * impulse
%  Make sure mConv is compiled.
% --Ayan Chakrabarti <ayanc@ttic.edu>
function [y,xz] = gmConv(x2,k2)

NZSTD=0.01;

nsamp = size(k2,3); ksz = size(k2,1);
ysz = size(x2,1)-ksz+1; crop = (ksz-1)/2;

k = gpuArray(k2); x = gpuArray(x2);
y = zeros([ysz*ysz nsamp],'single','gpuArray');
mConv;
y = y + randn(size(y),'single','gpuArray')*NZSTD;
y = max(0,min(1,y));


xz = gpuArray(x2(1+crop:end-crop,1+crop:end-crop,:));
xz = reshape(xz,[ysz*ysz nsamp]);