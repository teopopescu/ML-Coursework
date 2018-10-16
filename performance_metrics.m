function [one, two]=performance_metrics(Mdl, Xv,Yv, Posterior)
    %1)Accuracy to compare with academic paper: 86.58±1.78 for NB
    one=1-loss(Mdl,Xv,Yv,'LossFun','classiferror');
    %2)Cross Enthropy to compare RF & Naive Bayes
    %Step 1) Transform the original target vector to dummy vectors
    Yv2 = dummyvar(Yv);
    %Step 2) Apply the log to the output vector
    log_o=log(Posterior);
    %Step 3) Multiply the result from above with the actual target values
    product=log_o.*Yv2;
    %Step 4) Calculate cross entropy at row level
    row_e=sum(product,2);
    %Step 5) Calculate the mean cross entropy for the model
    two=mean(row_e);       
end