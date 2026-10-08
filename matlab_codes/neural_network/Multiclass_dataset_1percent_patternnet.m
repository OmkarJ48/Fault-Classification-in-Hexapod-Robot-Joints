% Step 1: Load the dataset
data = readtable('Hexapod_One_Joint_data_set.csv', 'VariableNamingRule', 'preserve');
disp(data.Properties.VariableNames);  % Display the names of the variables in the dataset
unique(data.Label);  % Display unique values in the Label column

% 2. Multiple Classifications (Fault Types) - Dataset 2
multiclassData = data;
multiclassData.Label = categorical(multiclassData.Label);  % Convert Label to categorical (fault types)

% Randomly select 1% of the data for the multiclass dataset
randomIndex = randperm(height(multiclassData), round(0.01 * height(multiclassData)));

% Create the multiclass dataset with 1% of random samples
multiclassData1percent = multiclassData(randomIndex, :);

% Save the multiclass  1% classification dataset
writetable(multiclassData1percent, 'multiclass_dataset_1%.csv');


% Step 1: Load the 1% multiclass dataset
data = readtable('multiclass_dataset_1%.csv', 'VariableNamingRule', 'preserve');
disp(data.Properties.VariableNames);  % Display the names of the variables in the dataset
unique(data.Label);  % Display unique values in the Label column

% Step 2: Convert labels to categorical (if not already)
multiclassData = data;  % Copy original data
multiclassData.Label = categorical(multiclassData.Label);  % Convert Label to categorical (fault types)

% Step 3: Separate features and labels
X = multiclassData{:, 1:end-1};  % Features (all columns except the last)
X = normalize(X);  % Normalize features
Y = multiclassData.Label;  % Labels (last column)

% Convert labels to one-hot encoding for patternnet
Y_onehot = onehotencode(Y, 2)';  % Convert to one-hot encoding (transpose to match input format)

% Step 4: Split the dataset into training (80%) and test (20%)
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);  % 80-20 split
trainIdx = training(cv);
testIdx = test(cv);

X_train = X(trainIdx, :)';  % Transpose to match the input format for patternnet
Y_train = Y_onehot(:, trainIdx);
X_test = X(testIdx, :)';
Y_test = Y_onehot(:, testIdx);

% Step 5: Define Neural Network
hiddenLayerSize = [15 10];  % Create the network with 10 neurons in the first hidden layer and 5 in the second layer
net = patternnet(hiddenLayerSize);  % Create the pattern recognition network

% Step 6: Set training parameters
net.trainParam.epochs = 200;  % Set epochs to 300 for training
net.trainParam.lr = 0.001;  % Set learning rate

% Step 7: Train the Neural Network
[net, tr] = train(net, X_train, Y_train);  % Train the network

% View the network
view(net);

% Step 8: Predict on Test Data
Y_pred = net(X_test);  % Get predicted probabilities

% Convert outputs to class labels
[~, Y_pred_class] = max(Y_pred, [], 1);  % Get predicted classes
[~, Y_true_class] = max(Y_test, [], 1);  % Get true classes

% Step 9: Confusion Matrix
figure;
confusionchart(Y_true_class, Y_pred_class);
title('Confusion Matrix for 1% Dataset');

% Step 10: Compute ROC Curve and AUC for a specific class (e.g., class 1)
class_to_evaluate = 1;  % Specify the class to evaluate
[Xroc, Yroc, ~, AUC] = perfcurve(Y_true_class, Y_pred(class_to_evaluate, :), class_to_evaluate);

% Plot ROC Curve for the specified class
figure;
plot(Xroc, Yroc);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve for Class: ' char(unique(multiclassData.Label(class_to_evaluate))) ' (AUC = ' num2str(AUC) ')']);
grid on;

trainPerf = tr.best_perf;
valPerf = tr.best_vperf;
testPerf = tr.best_tperf;

fprintf('Training Performance: %f\n', trainPerf);
fprintf('Validation Performance: %f\n', valPerf);
fprintf('Test Performance: %f\n', testPerf);