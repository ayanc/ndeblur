% function [imb,img,kgt,gt_mse] = loadSDB(iid,kid)
%  Load various images from the Sun 2013 benchmark. Please edit
%  this file to set the correct paths.
%
%  For image imgid (1-80), kernel id (1-8), returned values are:
%     imb: Blurry input image
%     img: Ground truth sharp image 
%          (cropped--smaller than imb by 100x100)
%     kgt: Ground truth kernel
%     gt_mse: MSE of EPLL estimate with ground truth kernel. 
% --Ayan Chakrabarti <ayanc@ttic.edu>
function [imb,img,kgt,gt_mse] = loadSDB(iid,kid)

base = '/home/ayan/data/deblur/';

% Blurry image
imb = im2single(imread([base 'input80imgs8kernels/' num2str(iid) '_' ...
		    num2str(kid) '_blurred.png']));

% Ground truth (cropped) image
img = im2double(imread([base 'all_deblur_results/img' num2str(iid) ...
		    '_groundtruth_img.png']));

% Ground truth kernel
kgt = im2single(imread([base 'all_deblur_results/kernel' num2str(kid) ...
		    '_groundtruth_kernel.png']));

% EPLL results with ground truth kernel
if nargout > 3
  imgbst = im2double(imread([base 'groundtruth_kernel_latent_zoran/kernel_' ...
		    num2str(kid) '/' num2str(iid) ...
		    '_gtk_latent_zoran.png']));
  gt_mse = getMSE(imgbst,img);
end;
