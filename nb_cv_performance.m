u% Get the mean performance metrics for all the folds
cv_time=endt-startt;
cv_mean_ac_cv=mean(ac_cv);
cv_mean_ce_cv=mean(ce_cv);
cv_mean_ac_cv;
cv_mean_ce_cv;        
%Save the results
accuracy=[accuracy;cv_mean_ac_cv];
cross_entropy=[cross_entropy;cv_mean_ce_cv];
time=[time;cv_time];