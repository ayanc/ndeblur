% Replace string references to layer names and types with actual
% indices and function handles.
%
% --Ayan Chakrabarti <ayanc@ttic.edu>
function nw = makeNWDef(nw_in, in_layers)

nw = nw_in;


s = struct;
for i = 1:length(in_layers)
  s = setfield(s,in_layers{i},-i);
end;


for i = 1:length(nw)
  
  nwi = nw{i};

    
  copy = 0; lnm = nwi{3};
  while isfield(s,lnm)
    copy = copy+1;
    lnm = sprintf('%s%d',[nwi{3} 'v'],copy);
  end;
  
  param = nwi{4};
  if length(param) == 0
    param = struct;
  end;

  ftype = nwi{1};
  param.fwd = eval(['@' ftype '_f']);
  param.bwd = eval(['@' ftype '_b']);
  
  
  bt_names = nwi{2};
  btms = [];
  fprintf('%10s : %10s %5d: <- ',lnm,ftype,i);
  for j = 1:length(bt_names)
    btmj = getfield(s,bt_names{j});
    fprintf('%d ',btmj);
    btms = [btms btmj];
  end;
  fprintf('\n');
  
  param.btms = btms;
  param.my_idx = i;
  
  s = setfield(s,lnm,0);
  s = setfield(s,nwi{3},i);
  
  nwi{3} = lnm;
  nw{i} = {lnm, param};
  
end;