function [Ktrain, Ktest, A] = sig_act(M_train, M_test, gamma)

N = size(M_train,2);  NN = size(M_test,2);

Dtrain = ((sum(M_train'.^2,2)*ones(1,N))+(sum(M_train'.^2,2)*ones(1,N))'-(2*(M_train'*M_train)));
Dtest = ((sum(M_train'.^2,2)*ones(1,NN))+(sum(M_test'.^2,2)*ones(1,N))'-(2*(M_train'*M_test)));

A = sqrt(mean(mean(Dtrain))); %% if you want to use this parameter then use sigmoid formula with sigma in Ktest

Ktrain = 0; %%% No use, just to make similar structure as kernel_rbf we use this
Ktest = 1./(1+exp(-Dtest)); 
