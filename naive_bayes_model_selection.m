%% Clear everything
clear all
close all
%% Read the numerical train dataset for Naive Bayes Model Selection
nb_ms=readtable('training_num80.csv');
%% specify that the last column is ordinal categorical data
avalues={'unacc','acc','good','vgood'};
nb_ms.acceptability=categorical(nb_ms.acceptability,avalues,'Ordinal',true);
%% Examine if any of the variables are highly correlated. 
CM=corrcoef(table2array(nb_ms(:,1:6)));
heatmap(CM);
% we can see none of the independent variables are highly correlated, so we
% can include them all. 
%% set the priors and the class names
tab=tabulate(nb_ms.acceptability); %calculate percentages
prior=cell2mat(transpose(tab(:,3)))/100; %turn them into right format
class_names={'unacc','acc','good','vgood'};
%% Hyper parameter optimization for fincnb
% this optimization can only optimize normal or kernel distributions. So we
% are going to pick the the most optimal values from it and compare it with
% a model which uses the multivariate multinomial distributions. 
%One of the variables that we are also going to explore is the number of
%folds
nb_initialize_results_collection_variables;
for f = 5:10
        %split predictors and target variables
        X=nb_ms(:,1:6);
        Y=nb_ms(:,7);
        rng(1);
        %set up the optimization function
        Mdl = fitcnb(X,Y,...
            'ClassNames',class_names,...
            'Prior',prior,...
            'OptimizeHyperparameters','all',...
            'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus',...
            'Kfold',f));
        % Get the best hyper parameters
        BestDist=Mdl.ModelParameters.DistributionNames;
        BestWidth=Mdl.ModelParameters.Width;
        BestKernelType=Mdl.ModelParameters.Kernel;        
        %save the results
        num_folds=[num_folds;f];
        best_dist=[best_dist;cellstr(BestDist)];
        best_width=[best_width;num2cell(BestWidth)];
        best_kernel=[best_kernel;cellstr(BestKernelType)];
end
%% Merge the results into 1 table
nb_hyperp_type_conversion;
normal_kernel_best_params=[num_folds best_dist best_width best_kernel];
%% We can see that for each CV type, we get slightly different results:
% Let's add the accuracy and cross entropy to each of the models above
nb_initialize_results_collection_variables;
for m=1:6
    %set up all the parameters
    nb_initialize_fold_results_collection_variables;
    distributions=char(table2cell(normal_kernel_best_params(m,2)));
    w=cell2mat(table2array(normal_kernel_best_params(m,3)));
    ktype=char(table2cell(normal_kernel_best_params(m,4)));
    num_folds = table2array(normal_kernel_best_params(m,1));
    training_err = zeros(num_folds,1);
    %start the CV folds
    rng(1);
    c = cvpartition(nb_ms.acceptability,'KFold',num_folds);
    startt=cputime;
    for fold = 1:num_folds
            nb_split_data;            
             % Train a model
                Mdl = fitcnb(Xtr,Ytr,...
                    'ClassNames',class_names,...
                    'Prior',prior,...
                    'DistributionNames',distributions,...
                    'Width',w,...
                    'Kernel',ktype);
            nb_predict_evaluate_test;
    end
        endt=cputime;
        nb_cv_performance;   
end; 
%% Add the results to the original table
nb_perfm_type_conversion;
normal_kernel_best_params=[normal_kernel_best_params accuracy cross_entropy time];
%% Compare the previous results with the 'mvmn' distribution across the same fold types
nb_initialize_results_collection_variables;
for f=5:10
    nb_initialize_fold_results_collection_variables;
    distributions='mvmn';
    training_err = zeros(f,1);
    %start the CV folds
    rng(1);
    c = cvpartition(nb_ms.acceptability,'KFold',f);
    startt=cputime;
    for fold = 1:f
        nb_split_data;
         % Train a model
            Mdl = fitcnb(Xtr,Ytr,...
                'ClassNames',class_names,...  
                'Prior',prior,...
                'DistributionNames',distributions);
         nb_predict_evaluate_test;
    end
    endt=cputime;
    nb_cv_performance;
    %Save the rest of parameters results. 
    num_folds=[num_folds;f];
    best_dist=[best_dist;cellstr(distributions)];
    best_kernel=[best_kernel;cellstr('NA')];
    best_width=[best_width;cellstr('NA')];        
end;
%% Save the results for the multinomial distributions
nb_hyperp_type_conversion;
nb_perfm_type_conversion;
multinomial_results=[num_folds best_dist best_width best_kernel accuracy cross_entropy time];
%% merge the two tables together for final model selection
final_nb_models = [normal_kernel_best_params; multinomial_results];
final_nb_models = sortrows(final_nb_models,6,{'descend'});
best_model=final_nb_models(1,:);
best_model;
writetable(final_nb_models,'final_nb_models.csv');
%% Train the model on the entire train set and make final predictions on the test set. 
% Read the numerical test dataset for Naive Bayes Model Selection
test_nb=readtable('test_num80.csv');
% specify that the last column is ordinal categorical data
test_nb.acceptability=categorical(test_nb.acceptability,avalues,'Ordinal',true);
%split predictors and target variables
Xtr=table2array(nb_ms(:,1:6));
Ytr=table2array(nb_ms(:,7));
Xv=table2array(test_nb(:,1:6));
Yv=table2array(test_nb(:,7));
%Train the model with kernel parameters
startt=cputime;
Mdl = fitcnb(Xtr,Ytr,...
            'ClassNames',class_names,...
            'Prior',prior,...
            'DistributionNames','kernel',...
            'Width',0.4293,...
            'Kernel','box');  
endt=cputime;
training_time=endt-startt;
% Predict Result on train set
[label,Posterior,Cost] = predict(Mdl,Xtr);
% Evaluate Model on train set
[ac_train_final, ce_train_final]=performance_metrics(Mdl, Xtr,Ytr, Posterior);

% save the results of predictions for test set
predictions=array2table([Ytr label]);
VarNames={'target','predictions'};
predictions.Properties.VariableNames = VarNames;
writetable(predictions,'final_results_train_nb.csv');

% Predict Result on test set
[label,Posterior,Cost] = predict(Mdl,Xv);
% Evaluate Model on test set
[ac_test_final, ce_test_final]=performance_metrics(Mdl, Xv,Yv, Posterior);

%% save the results of predictions for test set
predictions=array2table([Yv label]);
VarNames={'target','predictions'};
predictions.Properties.VariableNames = VarNames;
writetable(predictions,'final_results_nb.csv');



