TP = 3402;
FP = 0;
precision = TP / (TP + FP);
FN = 0;
recall = TP / (TP + FN);
F1_score = 2 * (precision * recall) / (precision + recall);
fprintf('F1-score: %.4f\n', F1_score);
if isnan(F1_score)
    F1_score = 0;
end
