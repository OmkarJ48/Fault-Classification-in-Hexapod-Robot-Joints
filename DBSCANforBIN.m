% Load data (Assuming 'binaryData' is a table with the last column as labels)
labels = binaryData{:, end};  % Extract labels (last column)
data1 = binaryData{:, 1:end-1};  % Extract features (all but last column)

% Apply SMOTE for Class Balancing
minority_class = 0;  % Adjust based on dataset
N = 50;  % Oversampling percentage
k = 5;  % Number of nearest neighbors
[data1, labels] = smote(data1, labels, minority_class, N, k);

% Get indices of faulty and non-faulty cases
fault_indices = find(labels == 1);  % Assuming 1 = 'fault'
nofault_indices = find(labels == 0);  % Assuming 0 = 'no fault'

% Undersample the majority class
undersampled_data = data1;
undersampled_labels = labels;

if length(nofault_indices) < length(fault_indices)
    num_to_sample = length(nofault_indices);  % Match minority class
    rand_indices = randperm(length(fault_indices), num_to_sample);
    undersampled_data = data1([nofault_indices; fault_indices(rand_indices)], :);
    undersampled_labels = labels([nofault_indices; fault_indices(rand_indices)]);
end

% Create new balanced dataset
balancedData = array2table(undersampled_data, 'VariableNames', binaryData.Properties.VariableNames(1:end-1));
balancedData.Label = undersampled_labels;  % Add back label column

% Normalize features using Z-score
features = balancedData{:, 1:end-1};  % Extract feature values
balanced1bin = zscore(features);  % Standardize data

% Feature Selection using Relief Algorithm
fprintf('Performing Feature Selection...\n');
[ranking, weights] = relieff(balanced1bin, balancedData.Label, 10);
num_features_to_select = ceil(size(balanced1bin, 2) * 0.7);  % Select top 70% features
selected_features = balanced1bin(:, ranking(1:num_features_to_select));

% Dimensionality Reduction (Optional: compare PCA and t-SNE)
fprintf('Performing Dimensionality Reduction...\n');
[coeff, score] = pca(selected_features);
reduced_data = score(:, 1:min(5, size(score, 2)));  % Select top 5 components

% Advanced Parameter Tuning for DBSCAN
epsilon_values = 0.05:0.05:1.0;  % Wider range
minPts_values = [10, 15, 20, 25, 30];
best_accuracy = 0;
best_params = struct('epsilon', 0, 'minPts', 0);

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
            classCounts = histcounts(balancedData.Label(predictedClusters == c), [0 1 2]);
            [~, maxClass] = max(classCounts);
            clusterMap(c) = maxClass - 1;
        end
        
        mappedPredictions = NaN(size(predictedClusters));
        validClusterIndices = predictedClusters > 0;
        mappedPredictions(validClusterIndices) = clusterMap(predictedClusters(validClusterIndices));
        
        % Compute accuracy
        confMat = confusionmat(balancedData.Label, mappedPredictions);
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
    classCounts = histcounts(balancedData.Label(predictedClusters == c), [0 1 2]);
    [~, maxClass] = max(classCounts);
    clusterMap(c) = maxClass - 1;
end

mappedPredictions = NaN(size(predictedClusters));
validClusterIndices = predictedClusters > 0;
mappedPredictions(validClusterIndices) = clusterMap(predictedClusters(validClusterIndices));

% Generate confusion matrix
confMat = confusionmat(balancedData.Label, mappedPredictions);

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



% Same SMOTE function as before (omitted for brevity)


% ----------- SMOTE Function -----------
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