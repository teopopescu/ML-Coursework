    

%% Read the numerical train dataset for Random Forest Model Selection
%% Clear everything
clear all
close all

%%Read the numerical dataset for Random Forest Model selection
train_set=readtable('training_num80.csv');

%%Specify that the last column is ordinal categorical data
categories={'unacc','acc','good','vgood'};
train_set.acceptability=categorical(train_set.acceptability,categories,'Ordinal',true);

%Hyperparameter setup for performing Grid Search
nb_of_trees=[10 20 40 50]; %Number of trees
max_num_splits=[100 200 50]; % Maximal number of decision splits
min_leaf_size =[2 3 4 5]; %Minimum number of observations per tree leaf
num_variables_to_sample=[2 4 6]; %Number of predictors to select at random for each split.

random_forest_initialize_hyperparameters;
for trees = nb_of_trees
    for nb_predictors = num_variables_to_sample
        for leaf_size = min_leaf_size
            for splits = max_num_splits
      
%Split predictors and target variables
                features=train_set(:,1:6);
                labels=train_set(:,7);
%Set up the optimization function and fit models   
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

%%Compute prediction and score matrix containing the probability of each observation originating from the class, computed as the fraction of observations of the class in a tree leaf.
                [Yfit,scores] = predict(Mdl,features);


% Evaluate model on the train set by computing its cross-entropy;
                [ce_train_rf]=cross_entropy(Mdl,table2array(labels), scores);

%Save the results into one table
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

%%Merge models' results into one table
random_forest_perform_type_conversion;
rf_models = [number_of_trees leafs number_of_splits number_of_predictors accuracy ce time]
rf_models = sortrows(rf_models,6 ,{'descend'});
writetable(rf_models, 'Random_Forest_Models.csv')

%Take the model with highes cross entropy loss and store it separately;
best_rf_model =rf_models(1,:);
writetable(best_rf_model, 'Best_Random_Forest_model.csv')

%Read the numerical test dataset for Random Forest Model selection
test_rf=readtable('test_num80.csv');

%Train again the best model and predict on train set, get predicted labels and save on a comparison file
final_training_start_time=cputime;
bestMdl =  TreeBagger(table2array(best_rf_model(1,1)), features,labels,...
    'Method','classification',...
    'ClassNames', categorical({'unacc','acc','good','vgood'}),...
    'MaxNumSplits', table2array(best_rf_model(1,3)), ...
    'NumVariablesToSample', table2array(best_rf_model(1,4)), ...
    'MinLeafSize', table2array(best_rf_model(1,2)), ...
    'OOBPrediction','on');
final_training_end_time=cputime;
final_train_time = final_training_end_time - final_training_start_time;

%Compute final model accuracy on train set
final_model_accuracy=(1-oobError(bestMdl,...
    'Mode','Ensemble'));

% Predict final results (labels and score matrix) on train set
[Yfit,scores] = predict(bestMdl,features);
  
% Save the results of predictions for train set
  predictions = [Yfit labels];
  VarNames = {'train_predictions','target'};
  predictions.Properties.VariableNames = VarNames;
  writetable(predictions,'final_results_train_Random_Forest.csv');

%%Predict Results on the test set
%Split test set into features and labels
test_features = test_rf(:,1:6);
test_labels = test_rf(:,7);

[test_Yfit,test_scores] = predict(bestMdl,test_features);

final_results = [test_labels test_Yfit];
VarNames = {'target','test_predictions'};
final_results.Properties.VariableNames = VarNames;
writetable(final_results,'Random_Forest_Final_Labels.csv');

%Compute final accuracy
err = error(bestMdl,test_features,test_Yfit);
accuracy_test_final= 1-mean(err);

%Confusion Matrix between target test labels and predicted test labels
confusion_matrix = confusionmat(table2array(test_labels),test_Yfit);
plotconfusion(categorical(table2array(test_labels)),categorical(test_Yfit));

%Compute final cross-entropy
test_rf.acceptability=categorical(test_rf.acceptability,categories,'Ordinal',true);
[ce_test_final] = cross_entropy(bestMdl,test_rf.acceptability, test_scores);



