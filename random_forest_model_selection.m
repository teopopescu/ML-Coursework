    
%% Read the numerical train dataset for Random Forest Model Selection

%read the data
train_rf=readtable('training_num80.csv');
test_rf=readtable('test_num80.csv');

%Train
[trainedClassifier, validationAccuracy] = trainRandomForestClassifier(train_rf)


%Feature importance
%https://www.kaggle.com/niklasdonges/end-to-end-project-with-python
%https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

% Cross-validaton


% Predict Results on the test set
test_features = table2array(test_nb(:,1:6));
test_labels = test_nb(:,7);
     
predicted_labels = trainedClassifier.predictFcn(test_rf) % <-- i don't think this is right. 

%calculate cross-entropy for the model on train set;

%Hyperparameter tuning: that means “adjust the settings to improve performance” (The settings are known as hyperparameters 
%to distinguish them from model parameters learned during training). The most common way to do this 
%is simply make a bunch of models with different settings, 
%evaluate them all on the same validation set, and see which one does best.

% Hyperparameter tuning; 
    %Grid search for hyperparameter optimization;also random search or bayesian optimization; 

%- criterion: The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
%min_samples_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
%min_samples_split 
%and n_estimators.

% display; number of trees in the forest; one for number of features to sample at random in each node; 
 
%Change the number of trees in the forest
%Change the maximum depth of the trees
%Change the size of the forest

% Save the results against their targets in a file
predictions= [test_labels array2table(predicted_labels)];
VarNames={'target','predictions'};
predictions.Properties.VariableNames = VarNames;
writetable(predictions,'final_results_random_forest.csv');

%Further Evaluation
    %Cross entropy and accuracy; 
    %Confusion Matrix
    %Precision and Recall
    %F-Score
    %Precision Recall Curve
    %ROC AUC Curve
    %ROC AUC Score

%delete training and test files after process is complete 


%Links
%https://towardsdatascience.com/demystifying-hyper-parameter-tuning-acb83af0258f
%https://www.kaggle.com/niklasdonges/end-to-end-project-with-python
%https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234
%https://towardsdatascience.com/understanding-hyperparameters-and-its-optimisation-techniques-f0debba07568
%https://towardsdatascience.com/cross-validation-70289113a072
%https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f
%https://towardsdatascience.com/demystifying-cross-entropy-e80e3ad54a8
%https://towardsdatascience.com/demystifying-entropy-f2c3221e2550
%https://stats.stackexchange.com/questions/344220/how-to-tune-hyperparameters-in-a-random-forest
%https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
%https://uk.mathworks.com/matlabcentral/answers/321200-random-forest-using-classification-learner-app
%https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234
%https://www.mathworks.com/matlabcentral/answers/151699-am-i-computing-cross-entropy-incorrectly
%https://www.jeremyjordan.me/hyperparameter-tuning/
%https://towardsdatascience.com/understanding-hyperparameters-and-its-optimisation-techniques-f0debba07568
%https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
