% Define the confusion matrix
TP = 134;  % True Positives
FP = 0;    % False Positives
FN = 0;    % False Negatives
TN = 10;   % True Negatives

% Calculate Precision and Recall
Precision = TP / (TP + FP);
Recall = TP / (TP + FN);

% Calculate F1-score
F1_score = 2 * (Precision * Recall) / (Precision + Recall);

% Display results
fprintf('Precision: %.2f\n', Precision);
fprintf('Recall: %.2f\n', Recall);
fprintf('F1-score: %.2f\n', F1_score);
