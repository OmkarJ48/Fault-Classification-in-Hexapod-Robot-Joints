fh2 = @(X, K2)(kmeans(X, K2));
eva2 = evalclusters(multinorm1, fh2, "CalinskiHarabasz", "KList", 2:5);
clear fh2

% Ensure eva2.OptimalK is correctly assigned
K2 = eva2.OptimalK;
disp("Optimal number of clusters: " + num2str(K2));  % Confirm that K2 is correct

% Get cluster indices based on the optimal number of clusters
clusterIndices2 = kmeans(multinorm1, K2);

% Display cluster evaluation criterion values
figure
bar(eva2.InspectedK, eva2.CriterionValues);
xticks(eva2.InspectedK);
xlabel("Number of clusters");
ylabel("Criterion values - Calinski-Harabasz Criterion");
legend("Optimal number of clusters is " + num2str(K2));
title("Evaluation of Optimal Number of Clusters");
clear eva2  % Clear eva2 after use

% Calculate centroids of the clusters
centroids2 = grpstats(multinorm1, clusterIndices2, "mean");

% Perform PCA for visualization
figure
[~, score2] = pca(multinorm1);

% Compute cluster means in PCA space
clusterMeans2 = grpstats(score2, clusterIndices2, "mean");

% Plot 2D scatter plot of the first two PCA components
h2 = gscatter(score2(:, 1), score2(:, 2), clusterIndices2);
for i3 = 1:numel(h2)
    h2(i3).DisplayName = strcat("Cluster", h2(i3).DisplayName);
end
clear h2 i3 score2
hold on
h2 = scatter(clusterMeans2(:, 1), clusterMeans2(:, 2), 50, "kx", "LineWidth", 2);
hold off
h2.DisplayName = "ClusterMeans";
clear h2 clusterMeans2
legend;
title("First 2 PCA Components of Clustered Data");
xlabel("First Principal Component");
ylabel("Second Principal Component");

% Assume you have true labels for comparison (trueLabels)
trueLabels = multiclassData.Label;  % Assuming this is your true label column

% Convert categorical labels to numeric if necessary
if iscategorical(trueLabels)
    trueLabels = double(trueLabels);  % Convert categorical to numeric
end

% Ensure trueLabels and predictedClusters have the same length
if length(trueLabels) ~= length(clusterIndices2)
    error("Mismatch: trueLabels and predictedClusters must have the same length.");
end

% Implementing weighted voting or a basic cluster label mapping
% Cluster Map (A basic example)
% Find the most frequent true label for each cluster
clusterMap = zeros(K2, 1); % Initialize clusterMap for K clusters
for i = 1:K2
    % Find indices of samples belonging to the current cluster
    clusterIndices = find(clusterIndices2 == i);
    
    % Get the true labels of those samples
    trueLabelsInCluster = trueLabels(clusterIndices);
    
    % Assign the most frequent true label to the cluster
    clusterMap(i) = mode(trueLabelsInCluster);
end

% Map the predicted clusters to true labels using the cluster map
mappedPredictions = arrayfun(@(x) clusterMap(x), clusterIndices2);

% Generate the confusion matrix
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