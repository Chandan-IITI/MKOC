
% Paper: Multi-layer Kernel Ridge Regression for One-class Classification
% Author(s):  Chandan Gautam, Alexandros Iosifidis
% Email: chandangautam31@gmail.com
% Institute: Discipline of CSE, IIT Indore

%%% Following lines of code evaluate the method MKOC_Thr2 i.e MKOC_Theta2 as per paper

clc;
clear all;
    
%load_data;
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels_all = loadMNISTLabels('train-labels.idx1-ubyte');
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels_all = loadMNISTLabels('t10k-labels.idx1-ubyte');

%%% Just keep in transpose format
train_data_all = train_images';
test_data_all = test_images';


%%% Convert test label into normal and outlier
digit_num = 5;
tedata_label_all = test_labels_all;
tedata_label_all(test_labels_all==digit_num)=1;
tedata_label_all(test_labels_all~=digit_num)=2;


%%% Divide train_data into normal and outlier samples
pos_data = train_data_all(train_labels_all==digit_num,:);
neg_data = train_data_all(train_labels_all~=digit_num,:);

tot_run = 5;
for run=1:tot_run  %%% Total number of time to rerun the experiment

%%% Preparing train,validation and test data

%%% load data from saved files for each run so same data can be used for
%%% other cassiifiers for fair experiment

%%% load saved files for each run
 load(['MNIST_subsample_digit_' num2str(digit_num) '_run_' num2str(run)])
    
    train_data = train_data_subsample';
    val_pos_ind = setdiff(1:size(pos_data,1),train_id);
    val_pos_data = pos_data(val_pos_ind,:);
    
    val_data = [val_pos_data_subsample;val_neg_data_subsample]';
    val_label = [ones(size(val_pos_data_subsample,1),1); ones(size(val_neg_data_subsample,1),1).*2];    
    
    %%% Preparing Test Data
    test_pos_data = test_data_all(tedata_label_all==1,:);
    test_neg_data = test_data_all(tedata_label_all==2,:);
    notest_samples = size(test_pos_data,1);
    test_data = [test_pos_data_subsample;test_neg_data_subsample]';
    test_labels = [ones(length(test_pos_id),1); ones(length(test_neg_id),1).*2];
    

   for layer=1:5
       
    %%% Normalization %%%
    %%%% One way
%     [ntrain_data norm_val] = norm_denorm(train_data', 1, 1);
%     ntest_data = norm_denorm(test_data', 1, 0, norm_val);
%     nval_data = norm_denorm(val_data', 1, 0, norm_val);

%%%   Another way
    mmod.nor.dat = mean_and_std(train_data', 'true');
    ntrain_data = normalize_data(train_data',mmod.nor.dat);
    ntest_data = normalize_data(test_data',mmod.nor.dat);
    nval_data = normalize_data(val_data',mmod.nor.dat); 
    
     clear train_data test_data val_data;
     train_data = ntrain_data'; test_data = ntest_data'; val_data = nval_data'; % Just for keeping in the required format
     clear ntrain_data ntest_data nval_data;
    %%% End of Normalization %%%
    
    noOfTrainData = size(train_data,2);
    
    % Range of all parameter
	                      graph_name = 'NO_Graph';
    Range_gamma = 0; %scale_range(train_data'); %%% RBF kernel Parameter
    Range_C = power(2,-5:5);
    temp_ind = 0;
    for i=1:length(Range_gamma)
        gama = Range_gamma(i);
        for j=1:length(Range_C)
            C = Range_C(j);     
     
                  % Any kernel function could be used
                  %%%% MKOC_Thr2
                  [labels_MKOC labeltr_MKOC] = MKOC_Thr2(train_data,val_data,layer,C,gama);   


                  % Concatenate training and validatin label as a final validation label
     act_val_lbls = val_label;
     pred_val_lbls = labels_MKOC;  
     
    temp_ind = temp_ind +1;        
    
       [run layer temp_ind]

    [accu(temp_ind) sens(temp_ind) spec(temp_ind) prec(temp_ind) rec(temp_ind) f11(temp_ind) gm(temp_ind)] = Evaluate(act_val_lbls,pred_val_lbls,1);
    whole_para(temp_ind,:) = [gama C];
                
                clear labels_MKOC labeltr_MKOC;
        end
    end

    %%%% Choose optimal parameters based on performance on validatin set
    [max_val opt_ind] = max(gm);
    gama_opt = whole_para(opt_ind, 1); 
    C_opt = whole_para(opt_ind, 2); 
    traccuracy(layer,run)=accu(opt_ind); trsensitivity(layer,run)=sens(opt_ind); trspecificity(layer,run)=spec(opt_ind);
    trprecision(layer,run)=prec(opt_ind); trrecall(layer,run)=rec(opt_ind); trf1(layer,run)=f11(opt_ind); trgmean(layer,run)=gm(opt_ind);
    clear gm;
    
    %%% Final training on optimal parameters
                  %%%% MKOC_Thr2
      [labels_MKOC, ~] = MKOC_Thr2(train_data, test_data, layer,C_opt,gama_opt); 

                    
    [accuracy(layer,run) sensitivity(layer,run) specificity(layer,run) precision(layer,run) recall(layer,run) f1(layer,run) gmean(layer,run)] = Evaluate(test_labels,labels_MKOC,1);
    

   end
   
end

xlswrite('MNIST_results_Thr2.xlsx', gmean.*100,['digit' num2str(digit_num)],'B2');

%%% For Testing
maccuracy = mean(accuracy,2)*100; msensitivity = mean(sensitivity,2)*100; mspecificity = mean(specificity,2)*100;
mprecision = mean(precision,2)*100; mrecall = mean(recall,2)*100; mf1 = mean(f1,2)*100; mgmean = mean(gmean,2)*100;

xlswrite('MNIST_results_Thr2.xlsx', mgmean,['digit' num2str(digit_num)],'H2');

gmean_final = gmean.*100;
stdev_gmean = std(gmean_final,0,2);
xlswrite('MNIST_results_Thr2.xlsx', stdev_gmean,['digit' num2str(digit_num)],'J2');

disp('   Layer1    Layer2    Layer3    Layer4    Layer5')
disp(mgmean')



