%To do: 
% create function for splitting data into test and train; X
% update this code on data cleaning;  X
% build model, fit, predict, evaluate; --needs improvement x
% hyperparameter tuning;
% build 2 visualizatons; 

% -------------------------Left to do; 
% Exploratory data analysis
% - descriptive statistics
   % 1 show stdev of data
   % 
  
% - descriptive visualizations 
    %  1 plot all variable distributions
    %  2 do feature selection (PCA, can use a decision tree as well) and
    %  display a scatter plot
    % Heatmap
    % k-means clustering visualization
    
    % Later:
    % visualize loss function differences for both Naive Bayes and Random
    % Forest
    
%% Read the numerical train dataset for Random Forest Model Selection


%read the data
train_rf=readtable('training_num80.csv');
test_rf=readtable('test_num80.csv');

%Train
[trainedClassifier, validationAccuracy] = trainRandomForestClassifier(train_rf)

 % Predict Results
test_features = table2array(test_nb(:,1:6));
test_labels = test_nb(:,7);
     
predicted_labels = trainedClassifier.predictFcn(test_rf) % <-- i don't think this is right. 
    
% Hyperparameter tuning; 


% - evaluate model and write some analysis ( here, compute accurracy errors) 


% Save the results against their targets in a file
predictions= [test_labels array2table(predicted_labels)];
VarNames={'target','predictions'};
predictions.Properties.VariableNames = VarNames;
writetable(predictions,'final_results_random_forest.csv');


%delete training and test files after process is complete 

