% Init network weights. Note that the size of the weights defines
% the number of units in each hidden layer.
% --Ayan Chakrabarti <ayanc@ttic.edu>
function wts = nw_wts

sz = 1024;

wts = cell(12,1);

wts{1} = {randn(sz,81)/sqrt(2*(81+208)), ...
	  randn(sz,208)/sqrt(2*(81+208)), ...
	  zeros(sz,1)};

for j = 2:3
  wts{j} = {randn(sz,208)/sqrt(2*(2*208)), ...
	    randn(sz,208)/sqrt(2*(2*208)), ...
	    zeros(sz,1)};
end;


for j = 4:5
  wts{j} = {randn(2*sz,sz)/sqrt(2*(2*sz)), ...
	    randn(2*sz,sz)/sqrt(2*(2*sz)), ...
	    zeros(2*sz,1)};
end;

sz = sz*2;

wts{6} = {randn(2*sz,sz)/sqrt(2*(2*sz)), ...
	  randn(2*sz,sz)/sqrt(2*(2*sz)), ...
	  zeros(2*sz,1)};

sz = sz*2;

wts{7} = {randn(sz,sz)/sqrt(2*sz), ...
	   zeros(sz,1)};

wts{8} = {randn(sz,sz)/sqrt(2*sz), ...
	   zeros(sz,1)};

wts{9} = {randn(sz,sz)/sqrt(2*sz), ...
	   zeros(sz,1)};

wts{10} = {randn(sz,sz)/sqrt(2*sz), ...
	   zeros(sz,1)};

wts{11} = {randn(4224,sz)/sqrt(sz), ...
	   zeros(4224,1)};

for i = 1:length(wts)
  for j = 1:length(wts{i})
    wts{i}{j} = single(wts{i}{j});
  end;
end;