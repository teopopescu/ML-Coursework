
function [outputArg1,outputArg2, outputArg3, outputArg4,outputArg5] = splitdata(inputArg1,inputArg2, inputArg3)

%This function takes 3 arguments: a filename, a type of partition
%and the test split ratio (e.g. 0.3 turning set into 70% training data and 30%
%test data and produces 4 files: 
%- a numerical training set
%- a numerical test set
%- a categorical training set
%- a categorical test set
original_data = readtable(inputArg1)

bvalues={'low','med','high','vhigh'};
original_data.buying=categorical(original_data.buying,bvalues,'Ordinal',true);

mvalues={'low','med','high','vhigh'};
original_data.maint=categorical(original_data.maint,mvalues,'Ordinal',true);

dvalues={'2','3','4','5more'};
original_data.doors=categorical(original_data.doors,dvalues,'Ordinal',true);

pvalues={'2','4','more'};
original_data.persons=categorical(original_data.persons,pvalues,'Ordinal',true);

lvalues={'small','med','big'};
original_data.lug_boot=categorical(original_data.lug_boot,lvalues,'Ordinal',true);

svalues={'low','med','high'};
original_data.safety=categorical(original_data.safety,svalues,'Ordinal',true);

avalues={'unacc','acc','good','vgood'};
original_data.acceptability=categorical(original_data.acceptability,avalues,'Ordinal',true);


rng(1);

cvpt = cvpartition(original_data.acceptability, inputArg2, inputArg3)
trainingIdx = training(cvpt)
testIdx = test(cvpt)

training_data = original_data(trainingIdx,:)
test_data = original_data(testIdx,:)

individual_variables = [double(original_data.buying) double(original_data.maint) double(original_data.doors) double(original_data.persons) double(original_data.lug_boot) double(original_data.safety)] ;
group = original_data.acceptability

VarNames={'buying','maint','doors','persons','lug_boot','safety','acceptability'};
numerical_original_data = [array2table(individual_variables) array2table(group)]
numerical_original_data.Properties.VariableNames = VarNames;

cvpt_2 = cvpartition(numerical_original_data.acceptability, inputArg2, inputArg3)
tr_idx = training(cvpt_2)
test_idx = test(cvpt_2)

training_data_num = numerical_original_data(tr_idx,:)
test_data_num = numerical_original_data(test_idx,:)

writetable(training_data_num, ['training_num' num2str(100-inputArg3*100) '.csv'])
writetable(test_data_num, ['test_num' num2str(100-inputArg3*100) '.csv'])
writetable(training_data,['training_cat' num2str(100-inputArg3*100) '.csv'])
writetable(test_data,['test_cat' num2str(100-inputArg3*100) '.csv'])


outputArg1 = training_data_num;
outputArg2 = test_data_num;
outputArg3 = training_data;
outputArg4 = test_data;
outputArg5 = original_data;


end


