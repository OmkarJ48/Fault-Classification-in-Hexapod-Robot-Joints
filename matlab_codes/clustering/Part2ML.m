% Assuming 'binaryDataSubset2' is a table with the labels in the last column
labels = binaryData{:, end};  % Assuming the label column is the last column
data1 = binaryData{:, 1:end-1};  % All features except the label

% Get the indices of faulty and non-faulty cases
fault_indices = find(labels == 1);  % Assuming 1 represents 'fault'
nofault_indices = find(labels == 0);  % Assuming 0 represents 'no fault'

% Undersample the majority class (faults)
undersampled_data = data1;
undersampled_labels = labels;

if length(nofault_indices) < length(fault_indices)
    % Majority class is 'fault', undersample it
    num_to_sample = length(nofault_indices);  % Match the number of samples in the minority class

    % Randomly select 'num_to_sample' samples from the majority class (faults)
    rand_indices = randperm(length(fault_indices), num_to_sample);
    undersampled_data = data1([nofault_indices; fault_indices(rand_indices)], :);
    undersampled_labels = labels([nofault_indices; fault_indices(rand_indices)]);
end

% Create the new balanced dataset
balancedData = array2table(undersampled_data, 'VariableNames', binaryDataSubset2.Properties.VariableNames(1:end-1));
balancedData.Label = undersampled_labels;  % Assuming 'Label' is the label column


% Exclude the Label column for normalization (we don't normalize labels)
feature_names = balancedData.Properties.VariableNames;
feature_names(strcmp(feature_names, 'Label')) = []; % Remove 'Label' from feature list

% Extract features (all columns except 'Label')
features = balancedData{:, feature_names};  % Extract numerical data for features

% Z-score Normalization (Standardization)
balanced1bin = zscore(features);  % Normalize the features using Z-score

% Convert the normalized features back into a table
normalized_data_table4 = array2table(balanced1bin, 'VariableNames', feature_names);

% Add the 'Label' column back to the normalized data table
normalized_data_table4.Label = balancedData.Label;

% Select optimal number of clusters (K value) using specified range
fh = @(X,K)(kmeans(X,K,"Distance","cosine","Replicates",20));
eva = evalclusters(balanced1bin,fh,"DaviesBouldin","KList",2:15);
clear fh
K = eva.OptimalK;
clusterIndices = eva.OptimalY;

% Display cluster evaluation criterion values
figure
bar(eva.InspectedK,eva.CriterionValues);
xticks(eva.InspectedK);
xlabel("Number of clusters");
ylabel("Criterion values - calinskiHarabaszCriterion");
legend("Optimal number of clusters is " + num2str(K))
title("Evaluation of Optimal Number of Clusters")
disp("Optimal number of clusters is " + num2str(K));
clear K eva

% Calculate centroids
centroids = grpstats(balanced1bin,clusterIndices,@(x)median(x,1));

% Display results

% Display 2D scatter plot (PCA)
figure
[~,score] = pca(balanced1bin);
clusterMeans = grpstats(score,clusterIndices,"mean");
h = gscatter(score(:,1),score(:,2),clusterIndices);
for i2 = 1:numel(h)
    h(i2).DisplayName = strcat("Cluster",h(i2).DisplayName);
end
clear h i2 score
hold on
h = scatter(clusterMeans(:,1),clusterMeans(:,2),50,"kx","LineWidth",2);
hold off
h.DisplayName = "ClusterMeans";
clear h clusterMeans
legend;
title("First 2 PCA Components of Clustered Data");
xlabel("First principal component");
ylabel("Second principal component");


% Ensure true labels and predicted clusters have the same number of elements
trueLabels = balancedData.Label;  % True labels
predictedClusters = clusterIndices;  % Clustering output

if length(trueLabels) ~= length(predictedClusters)
    error('Mismatch: trueLabels and predictedClusters must have the same number of elements.');
end

% Map clusters to actual class labels
numClusters = max(predictedClusters);  % Number of clusters found
clusterMap = zeros(numClusters, 1);  % Initialize mapping

for c = 1:numClusters
    % Find the most common true label in each cluster
    classCounts = histcounts(trueLabels(predictedClusters == c), [0 1 2]);  % For binary classes 0 and 1
    [~, maxClass] = max(classCounts);
    clusterMap(c) = maxClass - 1;  % Convert to 0 or 1
end

% Assign the best-matching labels to cluster indices
mappedPredictions = clusterMap(predictedClusters);

% Check if mappedPredictions is the correct size
if length(mappedPredictions) ~= length(trueLabels)
    error('Mismatch after mapping: mappedPredictions and trueLabels must have the same length.');
end

% Generate confusion matrix
confMat = confusionmat(trueLabels, mappedPredictions);

% Display Confusion Matrix
disp('Confusion Matrix:');
disp(confMat);

% Plot Confusion Matrix
figure
confusionchart(confMat);
title("Confusion Matrix of Clustering Performance");

% Compute clustering accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:));
fprintf('Clustering Accuracy: %.2f%%\n', accuracy * 100);