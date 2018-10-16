%% Clear everything
clear all
close all

%% Read the numerical train dataset for Naive Bayes Model Selection
nb_ms=readtable('trainNum.csv');

%% specify that the last column is ordinal categorical data
avalues={'unacc','acc','good','vgood'};
nb_ms.acceptability=categorical(nb_ms.acceptability,avalues,'Ordinal',true);

%% Summarize
summary(nb_ms);
tabulate(nb_ms.acceptability)

%% Cross Validation
% set out all the variables
ac_cv=[];
ce_cv=[];
prior = [0.6999 0.2220 0.0405 0.0376];
class_names={'unacc','acc','good','vgood'};
distributions={'mvmn','mvmn','mvmn','mvmn','mvmn','mvmn'};%multivariate multinomial distribution
num_folds = 10;
training_err = zeros(num_folds,1);

rng(1);
c = cvpartition(nb_ms.acceptability,'KFold',num_folds);

for fold = 1:10
    %get the train and validation sets
     train = nb_ms(c.training(fold), :);   
     val = nb_ms(c.test(fold), :);
     
     %split them into X and Y arrays
     Xtr=table2array(train(:,1:6));
     Ytr=table2array(train(:,7));

     Xv=table2array(val(:,1:6));
     Yv=table2array(val(:,7));
     
     % Train a model
        Mdl = fitcnb(Xtr,Ytr,...
            'ClassNames',class_names,...
            'Prior',prior,...
            'DistributionNames',distributions);
     
    % Predict Results
    [label,Posterior,Cost] = predict(Mdl,Xv);
    
    % Evaluate Model
    [ac, ce]=performance_metrics(Mdl, Xv,Yv, Posterior);
    ac_cv=[ac_cv;ac];
    ce_cv=[ce_cv;ce];
end

%% Get the mean performance metrics for all the folds
mean_ac_cv=mean(ac_cv);
mean_ce_cv=mean(ce_cv);

mean_ac_cv;
mean_ce_cv;


    










