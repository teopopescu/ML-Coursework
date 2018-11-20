    

%% Read the numerical train dataset for Random Forest Model Selection
%% Clear everything
clear all
close all

%read the numerical dataset for Random Forest Model selection
train_rf=readtable('training_num80.csv');

%% specify that the last column is ordinal categorical data
avalues={'unacc','acc','good','vgood'};
train_rf.acceptability=categorical(train_rf.acceptability,avalues,'Ordinal',true);


% do Random Forest yourself; 

%build all models in a nested for loop iterating through the hyperparameters;( nb of trees, depth )
%Mdl = TreeBagger....
%cross-validation 
%implement treeBagger after ON THE BEST MODEL taken from cross-validation;
%generate table with hyperparameters and out-of-bag error for each model

%setting a list of hyperparameters
nb_of_trees=[10 20 40 50];
max_depth=[10 5 15];
nb_of_splits=[100 200 50];
variables_at_random=[2 3 4 5];

rf_initialize_hyperparams;
for trees = nb_of_trees
    for depth = max_depth
        for splits = nb_of_splits
            for at_random = variables_at_random
%split predictors and target variables
        features=train_rf(:,1:6);
        labels=train_rf(:,7);
%set up the optimization function
    Mdl = TreeBagger(trees, features,labels,...
    'Method','classification',...
    'ClassNames', categorical({'acc'; 'good'; 'unacc'; 'vgood'}),...
    'MaxNumSplits', splits, ...
    'NumPredictorsToSample', at_random, ...  
    'OOBPrediction','on');

%No cross-validation needed as random forest chooses a bagged subset of input data at random
% setting NumPredictorsToSample to any value but 'all' enables TreeBagger to behave like RandomForest     
     %predict, store OOB error for each model
     
     
     %save the results into one table
     number_of_trees=[number_of_trees;trees];
     maximum_depth =[maximum_depth;depth];
     number_of_splits=[number_of_splits;splits];
     variables_to_sample = [ variables_to_sample;at_random];
     out_of_bag_error=[out_of_bag_error;oobError(Mdl)];

            end
        end
    end
    
end

% Get the model with the best accuracy from all models

%Take the best model and predict on test set; 

%read the numerical dataset for Random Forest Model selection
test_rf=readtable('test_num80.csv');


% Predict Results on the test set
test_features = table2array(test_nb(:,1:6));
test_labels = test_nb(:,7);

%Feature importance
%NO ! Principal Component Analysis (NB used corrcoef for analysis); <-- done on the trainedRandomForest;
% 2 models; one with all and one with 3-4 parameters; 
%have an optimized model 
%NO NEED for crossv-valid sa it's done on the for loop


% Save the results against their targets in a file
predictions= [test_labels array2table(predicted_labels)];
writetable(predictions,'final_results_random_forest.csv');

%Further Evaluation
    %accuracy; 
    %Confusion Matrix
    %Precision and Recall
    %F-Score
    %Precision Recall Curve
    %ROC AUC Curve
    %ROC AUC Score

%delete training and test files after process is complete 


