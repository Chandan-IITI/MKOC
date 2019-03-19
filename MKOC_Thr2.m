
% Paper: Multi-layer Kernel Ridge Regression for One-class Classification (MKOC)
% This code can also be teated as variant of Kernel Extreme Learning MAchine (KELM) and named as 
% Multi-layer Kernelized ELM for One-class Classification. Similarly, it can also be teated as variant
% of Lease Square SVM where bias is zero.
% Author(s):  Chandan Gautam, Alexandros Iosifidis
% Email: chandangautam31@gmail.com
% Institute: Discipline of CSE, IIT Indore

%%% Codes for MKOC_Thr2 i.e. MKOC_Theta2 as per paper

function [labels labeltr layerinfo] = MKOC_Thr1(train_data,test_data,max_layer,C,gamma)

for layer =1:max_layer
    if layer==1
        [K, Ktest, ~] = kernel_rbf(train_data,test_data,gamma);
        II = eye(size(K,1));

        if nargin < 6  
            B = K + II/C;
        else
            error('Provide proper number of Arguments for MKOC_Thr1');
        end
        if (layer~=max_layer)
            a = pinv(B) * train_data';
            Ot = a' * Ktest;
            Otr = a' * K;
        else
            Ot = test_data;
            Otr = train_data;
        end
    end

    if layer==max_layer
        
         clear B a K Ktest;
        [K, Ktest, ~] = kernel_rbf(Otr,Ot,gamma);
        II = eye(size(K,1));
      
        if nargin < 6  
            B = K + II/C;
        else
            error('Provide proper number of Arguments for MKOC_Thr1');
        end
        
        try
        a = pinv(B) * ones(size(K,2),1);
        catch
            warning('Problem with pinv, SVD did not converge');
        a = inv(B) * ones(size(K,2),1);
        end
        clear Ot Otr;
        Ot = a' * Ktest;
        Otr = a' * K;
    elseif ((layer~=max_layer) & (layer~=1))
        clear B a K Ktest;
        [K, Ktest, ~] = kernel_rbf(Otr,Ot,gamma);
        II = eye(size(K,1));
        
        if nargin < 6  
            B = K + II/C;
        else
            error('Provide proper number of Arguments for MKOC_Thr1');
        end
        try
            a = pinv(B) * Otr';
        catch
            warning('Problem with pinv, SVD did not converge');
            a = inv(B) * Otr';
        end
        clear Ot Otr;        
        Ot = a' * Ktest;
        Otr = a' * K;
    end
    
    %%%% Just gathering the information regarding outputweight of all layers and output
    %%%% of autoencoder at each layer except last layer as last is not an
    %%%% autoencoder but calculate final output.
    if layer==max_layer
        OutputWeight_max_layers= a';
    else
        OutputWeights(:,:,layer)= a';
        layer_autoenc_trains(:,:,layer)= Otr;
        layer_autoenc_tests(:,:,layer)= Ot;
    end
end

%%% Just storing all information into a single variable as a structre
layerinfo.OutputWeight_max_layer=OutputWeight_max_layers;
if (layer~=1)
layerinfo.OutputWeight=OutputWeights;
layerinfo.OutputWeight_max_layer=OutputWeight_max_layers;
layerinfo.layer_autoenc_train=layer_autoenc_trains;
layerinfo.layer_autoenc_test=layer_autoenc_tests;
end

%%% One-class Classification as per single node output %%%

%%%%% Start Thr2 (Theta2 as per paper)
%%% For training
meanOtr = mean(Otr);
thresh =meanOtr*0.05;%epsilon
labeltr = ones(size(K,2),1)*2;
difftr = abs(Otr-meanOtr);
labeltr(difftr<thresh)=1;

%%% For Testing
labels = ones(size(Ktest,2),1)*2;
%diffs = (Ot-meanOtr).^2; % or
diffs = abs(Ot-meanOtr);
labels(diffs<thresh)=1;
%%%%% End of Thr2 (Theta2 as per paper)

