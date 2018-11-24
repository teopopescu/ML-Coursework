    

%% Read the numerical train dataset for Random Forest Model Selection
%% Clear everything
clear all
close all

%read the numerical dataset for Random Forest Model selection
train_rf=readtable('training_num80.csv');

%% specify that the last column is ordinal categorical data
avalues={'unacc','acc','good','vgood'};
train_rf.acceptability=categorical(train_rf.acceptability,avalues,'Ordinal',true);

%generate table with hyperparameters and out-of-bag error for each model x

%setting a list of hyperparameters
nb_of_trees=[10 20 40 50];
max_num_splits=[100 200 50];
min_leaf_size =[2 3 4 5];
num_variables_to_sample=[2 4 6];

%Hyperparameters
%-nb of trees x
% -MaxNumSplits -Maximal number of decision splits 
%-MinLeafSize - minimum number of observations per tree leaf
%-NumVariablesToSample-Number of predictors to select at random for each split

rf_initialize_hyperparams;
for trees = nb_of_trees
    for nb_predictors = num_variables_to_sample
        for leaf_size = min_leaf_size
            for splits = max_num_splits
      
%split predictors and target variables
        features=train_rf(:,1:6);
        labels=train_rf(:,7);
%set up the optimization function
  %fit models;       
   start_time=cputime;
    Mdl = TreeBagger(trees, features,labels,...
    'Method','classification',...
    'ClassNames', categorical({'unacc','acc','good','vgood'}),...
    'MaxNumSplits', splits, ...
    'NumVariablesToSample', nb_predictors, ...
    'MinLeafSize', leaf_size, ...
    'SplitCriterion', 'deviance', ...
    'OOBPrediction','on');
end_time=cputime;
training_time=end_time-start_time;

%%predict, store OOB error for each model in a table
[Yfit,scores] = oobPredict(Mdl);

% Evaluate Model on train set
[ce_train_rf]=cross_ent(Mdl,table2array(labels), scores);

%No cross-validation needed as random forest chooses a bagged subset of input data at random
% setting NumPredictorsToSample to any value but 'all' enables TreeBagger to behave like RandomForest     
     
     %save the results into one table
     number_of_trees=[number_of_trees;trees];
     leafs=[leafs;leaf_size];
     number_of_splits=[number_of_splits;splits];
     number_of_predictors = [number_of_predictors;nb_predictors];
     accuracy=[accuracy; (1-oobError(Mdl,...
         'Mode','Ensemble'))];
     ce=[ce;ce_train_rf];
     time=[time;training_time];
            end
        end
    end
end

%%Merge results into one table
rf_perfm_type_conversion;
rf_models = [number_of_trees leafs number_of_splits number_of_predictors accuracy ce time]
rf_models = sortrows(rf_models,6 ,{'descend'});
writetable(rf_models, 'rf_models.csv')

%Take the model with highes cross entropy loss and store it separately;
best_rf_model =rf_models(1,:);
writetable(best_rf_model, 'best_rf_model.csv')


%read the numerical dataset for Random Forest Model selection
test_rf=readtable('test_num80.csv');

%Train again and predict on train set, get labels and save on file

final_training_start_time=cputime;
finalMdl =  TreeBagger(table2array(best_rf_model(1,1)), features,labels,...
    'Method','classification',...
    'ClassNames', categorical({'acc'; 'good'; 'unacc'; 'vgood'}),...
    'MaxNumSplits', table2array(best_rf_model(1,3)), ...
    'NumVariablesToSample', table2array(best_rf_model(1,4)), ...
    'MinLeafSize', table2array(best_rf_model(1,2)), ...
    'OOBPrediction','on');
final_training_end_time=cputime;
final_train_time = final_training_end_time - final_training_start_time;

%view(finalMdl.Trees{1},'Mode','graph')

%final model accuracy
final_model_accuracy=(1-oobError(finalMdl,...
    'Mode','Ensemble'));

% Predict final Result on train set
  [Yfit,scores] = oobPredict(finalMdl);
  
% save the results of predictions for train set
  predictions = [Yfit labels];
  VarNames = {'train_predictions','target'};
  predictions.Properties.VariableNames = VarNames;
  writetable(predictions,'final_results_train_Random_Forest.csv');

% Predict Results on the test set
test_features = table2array(test_rf(:,1:6));
test_labels = table2array(test_rf(:,7));

[test_Yfit,Cost] = predict(finalMdl, test_features);
final_results = array2table([test_labels test_Yfit]);
VarNames = {'target','test_predictions'};
final_results.Properties.VariableNames = VarNames;
writetable(final_results,'final_labels_Random_Forest.csv');

%Confusion Matrix

%confusion matrix between test labels and predictions
confusion_matrix = confusionmat(test_labels,test_Yfit);
cm = confusionchart(confusion_matrix);
%confusion_matrix_chart= confusionchart(categorical(test_labels),categorical(test_Yfit), ...
 %   'Title','My Title', ...
  %  'RowSummary','row-normalized', ...
   % 'ColumnSummary','column-normalized');

%%%% Automatic Hyperparameter Optimization


% Get the best hyper parameters


%%predict, store OOB error for each model in a table
%[Yfit,scores] = oobPredict(Mdl);

% Evaluate Model on train set
%[ce_train_rf]=cross_ent(Mdl,table2array(labels), scores);

%save the results into one table
 
   
%Future work: with increaed computational capacity, more hyperparameters could be tested;
 

