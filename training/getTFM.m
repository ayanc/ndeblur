% function tfm = getTFM
%
% Set up the different Fourier transforms for a 65x65 patch.
%
% tfm.inp{} from this function is going to be just the simple
% multi-scale decomposition. After calling this, you should apply a
% de-correlating transform to each tfm.inp{i}, so that the covariance
% matrix of [tfm.inp{i}*blurry_patch(:)] is the identity.
% 
% --Ayan Chakrabarti <ayanc@ttic.edu>
function tfm = getTFM

tfm = struct;

lpos = 33; % We'll be cropping center 33x33 patch output.
inp = cell(4,1);
f = mkFT(32,-1,4);
inp{1} = [ones(1,size(f,2))/65; f];

inp{2} = mkFT(32,4,8);
inp{3} = mkFT(16,4,8,32);
inp{4} = mkFT(8,4,8,32);

tfm.inp = inp;

tfm.full = mkFT(32,-1,Inf);

[x,y] = meshgrid([-32:32]);
tfm.lpos = find(abs(x) <= lpos & abs(y) <= lpos);

function fw = mkFT(p2,minF,maxF,pMax)

psz = 2*p2+1;

[xf,yf] = meshgrid([0:p2],[-p2:p2]);
xf = xf(:); yf = yf(:);
idx = find(xf == 0 & yf <= 0);
xf(idx) = []; yf(idx) = [];

idx = find( (abs(xf) > minF | abs(yf) > minF) & ...
	    abs(xf) <= maxF & abs(yf) <= maxF );
xf = xf(idx); yf = yf(idx);

xf = 2*xf*pi/psz; yf = 2*yf*pi/psz;

[x,y] = meshgrid([-p2:p2]);
x = x(:)'; y = y(:)';

xxf = bsxfun(@times,x,xf); yyf = bsxfun(@times,y,yf);

fw = zeros([length(xf)*2 length(x)]);
fw(1:2:end,:) = cos(xxf).*cos(yyf) - sin(xxf).*sin(yyf);
fw(2:2:end,:) = cos(xxf).*sin(yyf) + sin(xxf).*cos(yyf);

fw = sqrt(2)/psz*fw;

if exist('pMax')
  [x,y] = meshgrid([-pMax:pMax]);
  x = x(:); y = y(:);
  id = find(abs(x) <= p2 & abs(y) <= p2);
  fw0 = zeros(size(fw,1),length(x));
  fw0(:,id) = fw;
  fw = fw0;
end;

fw = single(fw);