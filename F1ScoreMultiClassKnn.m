function [precision, recall, f1_scores, weighted_f1] = compute_f1_scores(confMatrix)
    % Get the number of classes
    numClasses = size(confMatrix, 1);
    
    % Initialize precision, recall, and f1-score vectors
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1_scores = zeros(numClasses, 1);
    
    % Compute support (total instances per class)
    support = sum(confMatrix, 2);  % Sum of rows gives the total instances of each class
    total_samples = sum(support);  % Total number of instances in the dataset
    
    % Compute Precision, Recall, and F1-score for each class
    for i = 1:numClasses
        TP = confMatrix(i, i);               % True Positives
        FP = sum(confMatrix(:, i)) - TP;     % False Positives
        FN = sum(confMatrix(i, :)) - TP;     % False Negatives
        
        % Precision Calculation
        if TP + FP == 0
            precision(i) = 0;  % Avoid division by zero
        else
            precision(i) = TP / (TP + FP);
        end
        
        % Recall Calculation
        if TP + FN == 0
            recall(i) = 0;  % Avoid division by zero
        else
            recall(i) = TP / (TP + FN);
        end
        
        % F1-score Calculation
        if precision(i) + recall(i) == 0
            f1_scores(i) = 0;  % Avoid division by zero
        else
            f1_scores(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        end
    end
    
    % Compute Weighted F1-score
    weighted_f1 = sum((support / total_samples) .* f1_scores);

    % Display results
    disp('Class-wise Precision:');
    disp(precision);
    disp('Class-wise Recall:');
    disp(recall);
    disp('Class-wise F1-score:');
    disp(f1_scores);
    disp(['Weighted F1-score: ', num2str(weighted_f1)]);

end
confMatrix = [753 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
              2 755 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
              1 1 753 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
              2 1 2 740 2 9 0 0 0 0 0 0 0 0 0 0 0 0 0;
              1 0 0 2 755 1 0 0 0 0 0 0 0 0 0 0 0 0 0;
              1 0 0 9 1 755 1 0 0 0 0 0 0 0 0 0 0 0 0;
              0 0 0 0 0 1 755 1 0 0 0 0 0 0 0 0 0 0 0;
              0 0 0 0 0 0 1 755 1 0 0 0 0 0 0 0 0 0 0;
              0 0 0 0 0 0 0 1 754 2 0 0 0 0 0 0 0 0 0;
              0 0 0 0 0 0 0 0 1 753 1 0 0 0 0 0 0 0 0;
              0 0 0 0 0 0 0 0 0 1 756 0 0 0 0 0 0 0 0;
              0 0 0 0 0 0 0 0 0 0 0 756 0 0 0 0 0 0 0;
              0 0 0 0 0 0 0 0 0 0 0 0 755 1 0 0 0 0 0;
              0 0 0 0 0 0 0 0 0 0 0 0 1 754 1 0 0 0 0;
              0 0 0 0 0 0 0 0 0 0 0 0 0 1 755 1 0 0 0;
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 754 1 0 0;
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 755 1 0;
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 755 1;
              0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 755];

[precision, recall, f1_scores, weighted_f1] = compute_f1_scores(confMatrix);
