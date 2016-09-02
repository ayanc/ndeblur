% Do backward pass over network and compute all gradients
% --Ayan Chakrabarti <ayanc@ttic.edu>
function grads = doBackward(net,wts,activations,inputs,loss_list)

loss = num2cell(zeros(length(net),1,'single','gpuArray'));
for i = 1:length(loss_list)
  loss{loss_list{i}{1}} = loss_list{i}{2};
end;

grads = cell(length(net),1);

for i = length(net):-1:1
  [g_i,l_i] = net{i}{2}.bwd(net{i}{2},wts{i},activations,inputs,loss{i});
  grads{i} = g_i;
  
  btms = net{i}{2}.btms;
  for j = 1:length(btms)
    if btms(j) > 0
      loss{btms(j)} = loss{btms(j)} + l_i{j};
    end;
  end;
end;