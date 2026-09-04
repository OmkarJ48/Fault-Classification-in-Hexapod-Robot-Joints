clc; clear; close all;

% Load the datasets
binaryData = readtable('binary_classification.csv');
binaryData2 = readtable('binary_classification_2.csv');
multiclassData = readtable('multiclass_dataset.csv');
multiclassDataSubset = readtable('multiclass_classification.csv');

% Function to Compute and Plot ROC AUC
computeROC(binaryData, 'Binary Classification (Dataset 1)');
computeROC(binaryData2, 'Binary Classification (Dataset 2)');
computeROC(multiclassData, 'Multiclass Classification (Full Data)');
computeROC(multiclassDataSubset, 'Multiclass Classification (1% Subset)');

% Function to compute and plot ROC AUC
function computeROC(data, titleText)
    % Convert table to matrix
    X = table2array(data(:, 1:end-1)); % Features
    y = table2array(data(:, end)); % Labels
    
    % Convert categorical labels to numeric for multiclass
    if iscategorical(y)
        y = double(y); 
    end

    % Train-test split
    cv = cvpartition(y, 'HoldOut', 0.3);
    X_train = X(training(cv), :);
    y_train = y(training(cv));
    X_test = X(test(cv), :);
    y_test = y(test(cv));

    % Train classifier (Using SVM for binary, Ensemble for multiclass)
    if numel(unique(y_train)) == 2
        model = fitcsvm(X_train, y_train, 'Standardize', true, 'KernelFunction', 'linear');
    else
        model = fitcensemble(X_train, y_train, 'Method', 'Bag'); % Bagging Ensemble
    end

    % Get prediction scores
    [~, scores] = predict(model, X_test);

    % Compute ROC and AUC
    [X_roc, Y_roc, ~, AUC] = perfcurve(y_test, scores(:, end), max(y_test));

    % Plot ROC Curve
    figure;
    plot(X_roc, Y_roc, 'b-', 'LineWidth', 2);
    hold on;
    plot([0, 1], [0, 1], 'k--'); % Random classifier line
    xlabel('False Positive Rate (FPR)');
    ylabel('True Positive Rate (TPR)');
    title([titleText, ' - ROC Curve (AUC = ', num2str(AUC), ')']);
    grid on;
end
