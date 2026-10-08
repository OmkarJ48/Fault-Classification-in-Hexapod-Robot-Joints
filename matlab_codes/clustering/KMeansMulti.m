newTable = normalize(multiclassDataSubset(:,["X","Y","Z","slop(x)","slop(x/z)","slop(1/z)","slop(1/x)","slop(z/x)", ...
    "slop(z)"]));
numericData = table2array(newTable);
% Display results
f = figure("Units","Normalized");
dv = [1 2 3 4 5 6 7 8 9];
N = numel(dv);
f.Position = [0 0 1 N/3];
tiledlayout(N,2,"Padding","compact");
for k = 1:N
    nexttile
    plot(multiclassDataSubset.(dv(k)),"SeriesIndex",6, ...
        "DisplayName","Input data")
    ylabel(multiclassDataSubset.Properties.VariableNames{dv(k)},"Interpreter","none")
    if k == 1
        title("Input data")
    end

    nexttile
    plot(newTable.(k),"SeriesIndex",1,"LineWidth",1.5, ...
        "DisplayName","Normalized data")
    ylabel(multiclassDataSubset.Properties.VariableNames{dv(k)},"Interpreter","none")
    if k == 1
        title("Normalized data")
    end
end
f.NextPlot = "new";
clear f dv N k
% Select optimal number of clusters (K value) using specified range
fh = @(X,K)(kmeans(X,K,"Distance","sqeuclidean","Replicates",20));
eva = evalclusters(numericData, fh, "CalinskiHarabasz", "KList", 2:20);

% Retrieve optimal K value and cluster indices
K = eva.OptimalK;
clusterIndices = kmeans(numericData, K);

% Display cluster evaluation criterion values
figure
bar(eva.InspectedK, eva.CriterionValues);
xticks(eva.InspectedK);
xlabel("Number of clusters");
ylabel("Criterion values - Calinski-Harabasz Index");
legend("Optimal number of clusters: " + num2str(K))
title("Evaluation of Optimal Number of Clusters")
disp("Optimal number of clusters: " + num2str(K));

% Calculate centroids
centroids = grpstats(numericData, clusterIndices, "mean");

% Perform PCA for visualization
[coeff, score] = pca(numericData);

% Compute mean of clusters in PCA space
clusterMeans = grpstats(score(:, 1:2), clusterIndices, "mean");

% Display 2D scatter plot (PCA)
figure
h = gscatter(score(:, 1), score(:, 2), clusterIndices);
hold on
for i2 = 1:numel(h)
    h(i2).DisplayName = strcat("Cluster", h(i2).DisplayName);
end

% Plot cluster centroids
scatter(clusterMeans(:, 1), clusterMeans(:, 2), 50, "kx", "LineWidth", 2, "DisplayName", "Cluster Means");

hold off
legend;
title("First 2 PCA Components of Clustered Data");
xlabel("First Principal Component");
ylabel("Second Principal Component");

disp("Clustering complete and visualization generated.");
trueLabels = multiclassDataSubset.Label;
% Convert categorical labels to numeric if necessary
if iscategorical(trueLabels)
    trueLabels = double(trueLabels);  % Convert categorical to numeric
end

predictedClusters = clusterIndices;  % Clustering output

% Ensure true labels and predicted clusters have the same length
if length(trueLabels) ~= length(predictedClusters)
    error('Mismatch: trueLabels and predictedClusters must have the same number of elements.');
end

% Map clusters to actual class labels
numClusters = max(predictedClusters);  % Number of clusters found
clusterMap = zeros(numClusters, 1);  % Initialize mapping

for c = 1:numClusters
    % Find the most common true label in each cluster
    clusterMembers = trueLabels(predictedClusters == c); % Extract true labels for the cluster
    
    % Count occurrences of each label
    if isempty(clusterMembers)
        continue; % Skip empty clusters
    end
    
    classCounts = groupcounts(categorical(clusterMembers)); % Count unique labels
    [~, maxIdx] = max(classCounts); % Find the most common label
    
    uniqueClasses = unique(clusterMembers); % Get unique class values
    clusterMap(c) = uniqueClasses(maxIdx); % Assign the most frequent class label
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
