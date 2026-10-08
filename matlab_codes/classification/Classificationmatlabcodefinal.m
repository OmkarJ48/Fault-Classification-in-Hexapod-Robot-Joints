data = readtable('Hexapod_One_Joint_data_set.csv', 'VariableNamingRule', 'preserve');
disp(data.Properties.VariableNames);
unique(data.Label)
% converting data into binary and saving 
binaryData = data;
class(data.Label)  
binaryData = data;
binaryData.Label = double(binaryData.Label ~= 0); 
writetable(binaryData, 'binary_classification.csv');
summary(binaryData);

% 2. Multiple Classifications (Fault Types) - Dataset 2
multiclassData = data;
multiclassData.Label = categorical(multiclassData.Label);  % Convert Label to categorical (fault types)

% Randomly select 1% of the data for the multiclass dataset
randomIndex = randperm(height(multiclassData), round(0.01 * height(multiclassData)));

% Create the multiclass dataset with 1% of random samples
multiclassDataSubset = multiclassData(randomIndex, :);

% Save the multiclass classification dataset
writetable(multiclassDataSubset, 'multiclass_classification.csv');
writetable(multiclassData, 'multiclass_dataset.csv');
disp('Saved multiclass_classification.csv');

% 3. Binary Output (Fault/No Fault) - Dataset 3 (Another 1% subset)
% This is essentially the same as Dataset 1 but with a separate random sample
binaryDataSubset2 = binaryData(randperm(height(binaryData), round(0.01 * height(binaryData))), :);

% Save this second binary classification dataset
writetable(binaryDataSubset2, 'binary_classification_2.csv');
disp('Saved binary_classification_2.csv');
classificationLearner
