% Confusion matrix extracted from the image
conf_matrix = [11, 0, 0, 5, 0, 0, 0, 0, 0, 0;
               0, 5, 0, 7, 0, 0, 0, 0, 0, 0;
               0, 0, 7, 0, 8, 0, 0, 0, 0, 0;
               0, 0, 0, 8, 9, 0, 0, 0, 0, 0;
               0, 0, 0, 0, 11, 0, 0, 0, 0, 0;
               0, 0, 0, 0, 0, 11, 0, 0, 0, 2;
               0, 0, 0, 0, 0, 0, 5, 0, 0, 0;
               0, 0, 0, 0, 0, 0, 0, 7, 0, 0;
               0, 0, 0, 0, 0, 0, 0, 0, 6, 0;
               0, 0, 0, 0, 0, 0, 0, 0, 0, 9];

num_classes = size(conf_matrix, 1);
precision = zeros(1, num_classes);
recall = zeros(1, num_classes);
f1_scores = zeros(1, num_classes);

% Support (Total instances per class)
support = sum(conf_matrix, 2);
total_samples = sum(support);

for i = 1:num_classes
    TP = conf_matrix(i, i);  % True Positives
    FP = sum(conf_matrix(:, i)) - TP;  % False Positives
    FN = sum(conf_matrix(i, :)) - TP;  % False Negatives

    precision(i) = TP / (TP + FP + 1e-9);  % Avoid division by zero
    recall(i) = TP / (TP + FN + 1e-9);
    f1_scores(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + 1e-9);
end

% Compute Weighted F1-score
weighted_f1 = sum((support / total_samples) .* f1_scores);

% Display results
fprintf('Precision: \n');
disp(precision);
fprintf('Recall: \n');
disp(recall);
fprintf('F1 Scores: \n');
disp(f1_scores);
fprintf('Weighted F1-score: %.4f\n', weighted_f1);
