data = readtable('Hexapod_One_Joint_data_set.csv', 'VariableNamingRule', 'preserve');
disp(data.Properties.VariableNames);
unique(data.Label)
num_samples = height(data);
feature_names = data.Properties.VariableNames;
feature_names(strcmp(feature_names, 'Label')) = []; 
num_features = numel(feature_names);
classes = unique(data.Label);
num_classes = numel(classes);
num_samples_per_class = zeros(1, num_classes);
for i = 1:num_classes
    num_samples_per_class(i) = sum(data.Label == classes(i));
end
disp('Key Details of the Data:');
fprintf('Number of samples: %d\n', num_samples);
fprintf('Number of features: %d\n', num_features);
fprintf('Number of classes: %d\n', num_classes);
fprintf('Class names: ');
disp(classes);
fprintf('Number of samples per class:\n');
for i = 1:num_classes
    fprintf('Class %d: %d samples\n', classes(i), num_samples_per_class(i));
end
binaryData = data;
class(data.Label)  
binaryData = data;
binaryData.Label = double(binaryData.Label ~= 0); 
writetable(binaryData, 'binary_classification.csv');
summary(binaryData);
multiclassData = data;
multiclassData.Label = categorical(multiclassData.Label);  
randomIndex = randperm(height(multiclassData), round(0.01 * height(multiclassData)));
multiclassDataSubset = multiclassData(randomIndex, :);
writetable(multiclassDataSubset, 'multiclass_classification.csv');
writetable(multiclassData, 'multiclass_dataset.csv');
disp('Saved multiclass_classification.csv');
binaryDataSubset2 = binaryData(randperm(height(binaryData), round(0.01 * height(binaryData))), :);
writetable(binaryDataSubset2, 'binary_classification_2.csv');
disp('Saved binary_classification_2.csv');
num_faulty = sum(binaryData.Label == 1); 
num_no_fault = sum(binaryData.Label == 0); 
fprintf('Number of Faulty Cases: %d\n', num_faulty);
fprintf('Number of Non-Faulty Cases: %d\n', num_no_fault);
feature_names = binaryData.Properties.VariableNames;
feature_names(strcmp(feature_names, 'Label')) = []; 
features = binaryData{:, feature_names};  
binarynorm = zscore(features);  
normalized_data_table = array2table(binarynorm, 'VariableNames', feature_names);
normalized_data_table.Label = binaryData.Label;
writetable(normalized_data_table, 'normalized_binary_data.csv');
feature_names = binaryDataSubset2.Properties.VariableNames;
feature_names(strcmp(feature_names, 'Label')) = []; 
features = binaryDataSubset2{:, feature_names}; 
binary1norm = zscore(features);  
normalized_data_table1 = array2table(binary1norm, 'VariableNames', feature_names);
normalized_data_table1.Label = binaryDataSubset2.Label;
writetable(normalized_data_table1, 'normalized_binary_data1.csv');
feature_names = multiclassData.Properties.VariableNames;
feature_names(strcmp(feature_names, 'Label')) = [];
features = multiclassData{:, feature_names};  
multinorm1 = zscore(features); 
normalized_data_table2 = array2table(multinorm1, 'VariableNames', feature_names);
normalized_data_table2.Label = multiclassData.Label;
writetable(normalized_data_table2, 'normalized_multiclass.csv');
feature_names = multiclassDataSubset.Properties.VariableNames;
feature_names(strcmp(feature_names, 'Label')) = []; 
features = multiclassDataSubset{:, feature_names};  
multinorm2 = zscore(features); 
normalized_data_table3 = array2table(multinorm2, 'VariableNames', feature_names);
normalized_data_table3.Label = multiclassDataSubset.Label;
writetable(normalized_data_table3, 'normalized_multiclass1.csv');

