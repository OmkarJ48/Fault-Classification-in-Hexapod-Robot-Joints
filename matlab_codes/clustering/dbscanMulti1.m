% Load data (Assuming 'binaryData' is a table with the last column as labels)
labels3 = multiclassDataSubset{:, end};  % Extract labels (last column)
data4 = multiclassDataSubset{:, 1:end-1};  % Extract features (all but last column)

% Create new balanced dataset
balancedData4 = array2table(data4, 'VariableNames', binaryData.Properties.VariableNames(1:end-1));
balancedData4.Label = labels3;  % Add back label column

% Normalize features using Z-score
features = balancedData4{:, 1:end-1};  % Extract feature values
balancedmul1 = zscore(features);  % Standardize data

% Feature Selection using Relief Algorithm
fprintf('Performing Feature Selection...\n');
[ranking, weights] = relieff(balancedmul1, balancedData4.Label, 10);
num_features_to_select = ceil(size(balancedmul1, 2) * 0.7);  % Select top 70% features
selected_features = balancedmul1(:, ranking(1:num_features_to_select));

% Dimensionality Reduction (Optional: compare PCA and t-SNE)
fprintf('Performing Dimensionality Reduction...\n');
[coeff, score] = pca(selected_features);
reduced_data = score(:, 1:min(5, size(score, 2)));  % Select top 5 components

% Advanced Parameter Tuning for DBSCAN
epsilon_values =0.05:0.1:1.0;
minPts_values = [2,3,4,5,6,7,8,10,15,20];
best_accuracy = 0;
best_params = struct('epsilon', 0, 'minPts', 0);

% Ensure labels are numeric
original_labels = double(multiclassDataSubset{:, end});
unique_labels = unique(original_labels);

% Cross-validation like approach
fprintf('Tuning DBSCAN Parameters...\n');
for eps = epsilon_values
    for minPts = minPts_values
        % Apply DBSCAN
        clusterIndices = dbscan(reduced_data, eps, minPts);
        
        % Skip if no meaningful clusters
        if max(clusterIndices) <= 1
            continue;
        end
        
        % Map clusters to labels
        predictedClusters = clusterIndices;
        predictedClusters(predictedClusters == -1) = NaN;  % Handle noise
        
        % Majority voting for cluster labels
        numClusters = max(predictedClusters(~isnan(predictedClusters)));
        clusterMap = zeros(numClusters, 1);
        
        for c = 1:numClusters
            % Use numeric labels for accurate counting
            cluster_data = original_labels(predictedClusters == c);
            cluster_counts = accumarray(cluster_data, 1, [length(unique_labels), 1]);
            [~, maxClass] = max(cluster_counts);
            clusterMap(c) = unique_labels(maxClass);
        end
        
        mappedPredictions = NaN(size(predictedClusters));
        validClusterIndices = predictedClusters > 0;
        mappedPredictions(validClusterIndices) = clusterMap(predictedClusters(validClusterIndices));
        
        % Compute accuracy
        confMat = confusionmat(original_labels, mappedPredictions);
        accuracy = sum(diag(confMat)) / sum(confMat(:));
        
        % Update best parameters
        if accuracy > best_accuracy
            best_accuracy = accuracy;
            best_params.epsilon = eps;
            best_params.minPts = minPts;
        end
    end
end

% Final clustering with best parameters
fprintf('Best Parameters - Epsilon: %.2f, MinPts: %d\n', best_params.epsilon, best_params.minPts);
finalClusterIndices = dbscan(reduced_data, best_params.epsilon, best_params.minPts);

% Visualization
figure;
gscatter(reduced_data(:,1), reduced_data(:,2), finalClusterIndices);
title('Clustering Results after Advanced Preprocessing');
xlabel('First Reduced Dimension');
ylabel('Second Reduced Dimension');
legend;

% Confusion Matrix and Accuracy
predictedClusters = finalClusterIndices;
predictedClusters(predictedClusters == -1) = NaN;  % Handle noise

numClusters = max(predictedClusters(~isnan(predictedClusters)));
clusterMap = zeros(numClusters, 1);

for c = 1:numClusters
    % Use numeric labels for accurate counting
    cluster_data = original_labels(predictedClusters == c);
    cluster_counts = accumarray(cluster_data, 1, [length(unique_labels), 1]);
    [~, maxClass] = max(cluster_counts);
    clusterMap(c) = unique_labels(maxClass);
end

mappedPredictions = NaN(size(predictedClusters));
validClusterIndices = predictedClusters > 0;
mappedPredictions(validClusterIndices) = clusterMap(predictedClusters(validClusterIndices));

% Generate confusion matrix
confMat = confusionmat(original_labels, mappedPredictions);

% Display confusion matrix
disp('Confusion Matrix:');
disp(confMat);

% Plot Confusion Matrix
figure;
confusionchart(confMat);
title("Confusion Matrix of Advanced Clustering");

% Compute clustering accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:));
fprintf('Clustering Accuracy: %.2f%%\n', accuracy * 100);

