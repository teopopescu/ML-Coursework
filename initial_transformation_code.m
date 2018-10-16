%% Clear everything
clear all
close all

%% Read the original car evaluation dataset from https://archive.ics.uci.edu/ml/datasets/car+evaluation
odata = readtable('original_car_data.csv');

%% Investigate the data
summary(odata);

%All vectors are read as character vectors. We need to transform them into
%ordinal categorical vectors. 
bvalues={'low','med','high','vhigh'};
odata.buying=categorical(odata.buying,bvalues,'Ordinal',true);

mvalues={'low','med','high','vhigh'};
odata.maint=categorical(odata.maint,mvalues,'Ordinal',true);

dvalues={'2','3','4','5more'};
odata.doors=categorical(odata.doors,dvalues,'Ordinal',true);

pvalues={'2','4','more'};
odata.persons=categorical(odata.persons,pvalues,'Ordinal',true);

lvalues={'small','med','big'};
odata.lug_boot=categorical(odata.lug_boot,lvalues,'Ordinal',true);

svalues={'low','med','high'};
odata.safety=categorical(odata.safety,svalues,'Ordinal',true);

avalues={'unacc','acc','good','vgood'};
odata.acceptability=categorical(odata.acceptability,avalues,'Ordinal',true);

summary(odata)

%% check unique rows 
height(unique(odata))

%% Exploratory Data Analysis
% in order to do a parrallel coordinate plot, first we need to convert the independent variables to
% numeric equivalents
indv=[double(odata.buying) double(odata.maint) double(odata.doors) double(odata.persons) double(odata.lug_boot) double(odata.safety)] ;
group=odata.acceptability;
labels={'buying','maint','doors','persons','lug_boot','safety'};
parallelcoords(indv,'Group',group,'Labels',labels,'LineWidth',2);

% We can see from the graph that the variables that seem to drive the
% unacceptability are a low number of people and low safety, high mantenance and buying. The variables
% that seem to drive the v good acceptability are low buying price, high
% persons, high lugage boot and high safety. 

%% Means by Group
numeric_table=[array2table(indv) array2table(group)];
VarNames={'buying','maint','doors','persons','lug_boot','safety','acceptability'};
numeric_table.Properties.VariableNames = VarNames;

means = grpstats(numeric_table,'acceptability');

indv_means=table2array(means(:,3:8));
labels=means(:,3:8).Properties.VariableNames;
labels=labels';

%% spider plot
% % Axes properties 
 axes_interval = 2; 
 axes_precision = 1;  
% % Spider plot 
 spider_plot(indv_means, labels, axes_interval, axes_precision,... 
 'Marker', 'o',... 
 'LineStyle', '-',... 
 'LineWidth', 2,... 
 'MarkerSize', 5); 
% 
% % Title properties 
 title('Sample Spider Plot',... 
 'Fontweight', 'bold',... 
 'FontSize', 12); 
% 
% % Legend properties 
legend_values={'unacc','acc','good','vgood'};
legend('show', 'Location', 'southoutside',legend_values);
 
%% Split the categorical dataset into train and test. The train set is going to be used for model selection for both random forest and naive bayes. The test set is going to be used to compare the final 2 models and it will not be used during the model selection phase. 
%categorical
rng('default');
First_split = cvpartition(odata.acceptability,'Holdout',0.2);
trainCat = odata(training(First_split),:);
testCat = odata(test(First_split),:);

%numerical - since the independent variables are ordinal values, we can
%replace them by numbers???? need to double check the impact
rng('default');
First_split = cvpartition(numeric_table.acceptability,'Holdout',0.2);
trainNum = numeric_table(training(First_split),:);
testNum = numeric_table(test(First_split),:);

%% Save to CSV
writetable(trainCat,'trainCat.csv');
writetable(testCat,'testCat.csv');
writetable(trainNum,'trainNum.csv');
writetable(testNum,'testNum.csv');








