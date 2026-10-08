% Select optimal number of clusters (K value) using specified range
fh = @(X,K)(kmeans(X,K,"Distance","correlation"));
eva = evalclusters(binarynorm,fh,"CalinskiHarabasz","KList",2:5);
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
transformedData = binarynorm-mean(binarynorm,2);
transformedData = transformedData./sqrt(sum(transformedData.^2,2));
centroids = grpstats(transformedData,clusterIndices,"mean");
clear transformedData

% Display results

% Display 2D scatter plot (PCA)
figure
[~,score] = pca(binarynorm);
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
trueLabels = binaryData.Label;  % True labels
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