% Assuming 'binaryDataSubset2' is a table with the labels in the last column
labels = binaryDataSubset2{:, end};  % Assuming the label column is the last column
data2 = binaryDataSubset2{:, 1:end-1};  % All features except the label

% Apply SMOTE for Class Balancing
minority_class = 0;  % Adjust based on your dataset
N = 200;  % Oversampling percentage (adjust as needed)
k = 5;    % Number of nearest neighbors
[data2, labels] = smote(data2, labels, minority_class, N, k);

% Get the indices of faulty and non-faulty cases
fault_indices = find(labels == 1);  % Assuming 1 represents 'fault'
nofault_indices = find(labels == 0);  % Assuming 0 represents 'no fault'

% Undersample the majority class (faults)
undersampled_data = data2;
undersampled_labels = labels;

if length(nofault_indices) < length(fault_indices)
    % Majority class is 'fault', undersample it
    num_to_sample = length(nofault_indices);  % Match the number of samples in the minority class

    % Randomly select 'num_to_sample' samples from the majority class (faults)
    rand_indices = randperm(length(fault_indices), num_to_sample);
    undersampled_data = data2([nofault_indices; fault_indices(rand_indices)], :);
    undersampled_labels = labels([nofault_indices; fault_indices(rand_indices)]);
end

% Create the new balanced dataset
balancedData2 = array2table(undersampled_data, 'VariableNames', binaryDataSubset2.Properties.VariableNames(1:end-1));
balancedData2.Label = undersampled_labels;  % Assuming 'Label' is the label column

% Exclude the Label column for normalization (we don't normalize labels)
feature_names = balancedData2.Properties.VariableNames;
feature_names(strcmp(feature_names, 'Label')) = []; % Remove 'Label' from feature list

% Extract features (all columns except 'Label')
features = balancedData2{:, feature_names};  % Extract numerical data for features

% Z-score Normalization (Standardization)
balanced2bin = zscore(features);  % Normalize the features using Z-score

% Convert the normalized features back into a table
normalized_data_table4 = array2table(balanced2bin, 'VariableNames', feature_names);

% Add the 'Label' column back to the normalized data table
normalized_data_table4.Label = balancedData2.Label;

% Select optimal number of clusters (K value) using specified range
fh = @(X,K)(kmeans(X,K,"Distance","cityblock","Replicates",20));
eva = evalclusters(balanced2bin,fh,"DaviesBouldin","KList",1:15);
clear fh
K = eva.OptimalK;
clusterIndices = eva.OptimalY;

% Display cluster evaluation criterion values
figure
bar(eva.InspectedK,eva.CriterionValues);
xticks(eva.InspectedK);
xlabel("Number of clusters");
ylabel("Criterion values - silhouette");
legend("Optimal number of clusters is " + num2str(K))
title("Evaluation of Optimal Number of Clusters")
disp("Optimal number of clusters is " + num2str(K));
clear K eva

% Add this after you compute the cluster indices
% Compute VRC for each K in the range (assuming evalclusters has been used)

% List of K values that you evaluated
K_values = 1:15;  % You specified this range in evalclusters

% Initialize array to store VRC for each K value
VRC_values = zeros(1, length(K_values));

% Loop over each K value and compute VRC
for K_idx = 1:length(K_values)
    currentK = K_values(K_idx);
    
    % Perform k-means clustering for the current K value
    currentClusterIndices = kmeans(balanced2bin, currentK, 'Distance', 'cityblock', 'Replicates', 20);
    
    % Compute VRC for the current clustering result
    VRC_values(K_idx) = computeVRC(balanced2bin, currentClusterIndices);
end

% Display VRC values for each K
figure
bar(K_values, VRC_values);
xticks(K_values);
xlabel('Number of Clusters (K)');
ylabel('Variance Ratio Criterion (VRC)');
title('VRC for Different Numbers of Clusters');


% Calculate centroids
centroids = grpstats(balanced2bin,clusterIndices,@(x)median(x,1));

% Display results

% Display 2D scatter plot (PCA)
figure
[~,score] = pca(balanced2bin);
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
trueLabels = balancedData2.Label;  % True labels
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

% SMOTE Function Implementation
function [X_SMOTE, Y_SMOTE] = smote(X, Y, minority_class, N, k)
    X_minority = X(Y == minority_class, :);
    num_minority = size(X_minority, 1);
    num_synthetic = round((N / 100) * num_minority);
    
    neighbors = knnsearch(X_minority, X_minority, 'K', k+1);
    neighbors(:,1) = [];
    
    X_synthetic = zeros(num_synthetic, size(X, 2));
    for i = 1:num_synthetic
        idx = randi(num_minority);
        neighbor_idx = neighbors(idx, randi(k));
        diff = X_minority(neighbor_idx, :) - X_minority(idx, :);
        gap = rand();
        X_synthetic(i, :) = X_minority(idx, :) + gap * diff;
    end
    
    X_SMOTE = [X; X_synthetic];
    Y_SMOTE = [Y; repmat(minority_class, num_synthetic, 1)];
end

function VRC = computeVRC(data, labels)
    uniqueClusters = unique(labels);
    uniqueClusters(uniqueClusters == -1) = []; % Remove noise points (-1)
    numClusters = numel(uniqueClusters);

    % Compute centroids of each cluster
    centroids = zeros(numClusters, size(data, 2));
    for i = 1:numClusters
        centroids(i, :) = mean(data(labels == uniqueClusters(i), :), 1);
    end

    % Compute scatter within each cluster (variance of data points within each cluster)
    scatterWithin = zeros(numClusters, 1);
    for i = 1:numClusters
        clusterData = data(labels == uniqueClusters(i), :);
        scatterWithin(i) = mean(vecnorm(clusterData - centroids(i, :), 2, 2).^2);
    end

    % Compute scatter between clusters (variance between cluster centroids)
    scatterBetween = 0;
    globalMean = mean(data, 1);
    for i = 1:numClusters
        scatterBetween = scatterBetween + sum(vecnorm(centroids(i, :) - globalMean, 2, 2).^2) * sum(labels == uniqueClusters(i));
    end

    % Compute Variance Ratio Criterion
    VRC = scatterBetween / sum(scatterWithin);
end

