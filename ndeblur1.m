%function im_n = ndeblur1(imb,model)
%
% Stage 1 of algorithm: Compute direct neural average estimate im_n
% of sharp image, given blurry input imb. model contains the
% trained network model, returned as model = load('model.mat');
%
% --Ayan Chakrabarti <ayanc@ttic.edu>
function im_n = ndeblur1(imb,model)

im_n = pforward(imb,model.tfm,model.net,model.wts);
