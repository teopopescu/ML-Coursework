%get the train and validation sets
train = nb_ms(c.training(fold), :);   
val = nb_ms(c.test(fold), :);
%split predictors and target variables
Xtr=table2array(train(:,1:6));
Ytr=table2array(train(:,7));
Xv=table2array(val(:,1:6));
Yv=table2array(val(:,7));