%function [im,k] = ndeblur2(imb,im_n)
%
% Stage 2 of algorithm: Given output of stage 1 im_n, and the
%   original input imb, estimate global kernel k, and do blind
%   deconvolution with EPLL to get final estimate of sharp image
%   im.
%
% Note that you should have downloaded the EPLL code from Zoran &
% Weiss, and added it to your MATLAB path prior to calling this
% function.
%
% --Ayan Chakrabarti <ayanc@ttic.edu>
function [im,k] = ndeblur2(imb,imn)

KSZ=51; NSD=0.01;

tm=tic; [k,kerr] = estK(imb,imn,KSZ);tm = toc(tm);
fprintf('KERR = %2e\n',kerr);
fprintf('Kernel estimation took %.2f secs, starting EPLL\n',tm);
im = doEPLL(imb,k,max(NSD,kerr));
