function sX = cmptSdtParams2(X, Xm, Xstd)

Nx = size(X, 2); % number of training points

% standardize training set
sX = ( X- Xm*ones(1,Nx) )./( Xstd*ones(1,Nx) );
