%% Clear everything
clear all
close all
%% Create train and test data
[train_num,test_num,train_cat,test_cat,odata] = splitdata('original_car_data.csv','HoldOut',0.2)
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
labels={'buying','maint','doors','persons','lug_boot','safety'};
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
 title('Average Value per Predictor by Target Class',... 
 'Fontweight', 'bold',... 
 'FontSize', 12); 
% 
% % Legend properties 
legend_values={'unacc','acc','good','vgood'};
legend('show', 'Location', 'southoutside',legend_values); 

%%
figure(1)
for i=1:6
    [tbl,chi2,p,labels]=crosstab(table2array(odata(:,i)),odata.acceptability);
    rowNames = transpose(labels(:,1));
    colNames = transpose(labels(:,2));
    subplot(2,3,i);
    bar(tbl,'stacked')
    xticklabels(rowNames);
    %legend(colNames,'Location','northwestoutside');
    title(odata.Properties.VariableNames(i));
    hold on
end
%% target variable viz
histogram(odata.acceptability);









