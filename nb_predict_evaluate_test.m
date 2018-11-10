% Predict Results
[label,Posterior,Cost] = predict(Mdl,Xv);
% Store the results against their targets
target_all=[target_all; Yv];
predictions_all=[predictions_all;label];
% Evaluate Model
[ac, ce]=performance_metrics(Mdl, Xv,Yv, Posterior);
ac_cv=[ac_cv;ac];
ce_cv=[ce_cv;ce];