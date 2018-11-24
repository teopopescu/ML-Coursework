function [ce] = cross_ent(Mdl,Yv,Posterior)
%UNTITLED Summary of this function goes here
% %2) Cross entropy to compare RF and NB
    %Step 1) Transform the original target vector to dummy vectors
    Yv2 = dummyvar(Yv);
    %Step 2) Replace the '0' in posterior probabilities with 0.000001 so
    %that the log conversion is not infinite. 
    Posterior2=Posterior;
    Posterior2(Posterior2 == 0)=0.00001;
    %Step 3) Apply the log to the output vector
    log_o=log(Posterior2);
    %Step 4) Multiply the result from above with the actual target values
    product=log_o.*Yv2;
    %Step 5) Calculate cross entropy at row level
    row_e=sum(product,2);
    %Step 6) Calculate the mean cross entropy for the model
    ce=mean(row_e);      
end

