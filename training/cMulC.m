% function z = cMulC(x,y)
%   z = x y^*
%
% --Ayan Chakrabarti <ayanc@ttic.edu>
function z = cMulC(x,y)

z = zeros(size(x),'single','gpuArray');
z(1:2:end) = x(1:2:end).*y(1:2:end) + x(2:2:end).*y(2:2:end);
z(2:2:end) = -x(1:2:end).*y(2:2:end) + x(2:2:end).*y(1:2:end);
