
% Step 1: Load the dataset
data = readtable('Hexapod_One_Joint_data_set.csv', 'VariableNamingRule', 'preserve');
disp(data.Properties.VariableNames);  % Display the names of the variables in the dataset
unique(data.Label);  % Display unique values in the Label column

% Step 2: Convert labels to categorical
class(data.Label);
multiclassData = data;  % Copy original data
multiclassData.Label = categorical(multiclassData.Label);  % Convert Label to categorical (fault types)

% Step 3: Save the dataset (optional)
writetable(multiclassData, 'multiclass_dataset.csv');
summary(multiclassData);  % Display summary of the dataset

% Load the multiclass dataset
data = readtable('multiclass_dataset.csv', 'VariableNamingRule', 'preserve');

% Step 4: Separate features and labels
X = data{:, 1:end-1};  % Features (all columns except the last)
X = normalize(X);  % Normalize features
Y = data.Label;  % Labels (last column)

% Convert labels to categorical for pattern recognition
Y = categorical(Y);  % Convert to categorical format
tabulate(Y);  % Shows how many samples per class

% Convert labels to one-hot encoding for patternnet
Y_onehot = onehotencode(Y, 2)';  % Convert to one-hot encoding (transpose to match input format)

% Step 5: Split the dataset into training (80%) and test (20%)
cv = cvpartition(size(X,1), 'HoldOut', 0.2);  % 80-20 split
trainIdx = training(cv);
testIdx = test(cv);

X_train = X(trainIdx, :)';  % Transpose to match the input format for patternnet
Y_train = Y_onehot(:, trainIdx);
X_test = X(testIdx, :)';
Y_test = Y_onehot(:, testIdx);

% Step 6: Define Neural Network
hiddenLayerSize = [10, 5];  % Create the network with 10 neurons in the first hidden layer and 5 in the second layer
net = patternnet(hiddenLayerSize);  % Create the pattern recognition network

% Step 7: Set training parameters
net.trainParam.epochs = 300;  % Set epochs to 300 for training
net.trainParam.lr = 0.0001;  % Set learning rate

% Step 8: Train the Neural Network
[net, tr] = train(net, X_train, Y_train);  % Train the network

% View the network
view(net);

% Step 9: Predict on Test Data
Y_pred = net(X_test);  % Get predicted probabilities

% Convert outputs to class labels
[~, Y_pred_class] = max(Y_pred, [], 1);  % Get predicted classes
[~, Y_true_class] = max(Y_test, [], 1);  % Get true classes

% Step 10: Confusion Matrix
figure;
confusionchart(Y_true_class, Y_pred_class);
title('Confusion Matrix');

% Step 11: Compute ROC Curve and AUC for a specific class (e.g., class 1)
class_to_evaluate = 1;  % Specify the class to evaluate
[Xroc, Yroc, ~, AUC] = perfcurve(Y_true_class, Y_pred(class_to_evaluate, :), class_to_evaluate);

% Plot ROC Curve for the specified class
figure;
plot(Xroc, Yroc);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve for Class: ' char(unique(data.Label(class_to_evaluate))) ' (AUC = ' num2str(AUC) ')']);
grid on;


trainPerf = tr.best_perf;
valPerf = tr.best_vperf;
testPerf = tr.best_tperf;

fprintf('Training Performance: %f\n', trainPerf);
fprintf('Validation Performance: %f\n', valPerf);
fprintf('Test Performance: %f\n', testPerf);