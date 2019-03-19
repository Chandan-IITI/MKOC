function [sX, Xm, Xstd] = cmptSdtParams(X)

Nx = size(X, 2); % number of training points

% compute feature mean from training set
Xm = mean(X, 2); % total mean

% compute feature standard deviation from training set
Xstd = std(X, 0, 2);

% if std of a feature is 0 then do not divide it
Xstd(Xstd == 0) = 1;

% standardize training set
sX = ( X- Xm*ones(1,Nx) )./( Xstd*ones(1,Nx) );
