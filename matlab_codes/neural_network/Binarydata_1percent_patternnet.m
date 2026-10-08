%% Step 1: Load the dataset
data = readtable('binary_dataset_1%.csv', 'VariableNamingRule', 'preserve');
disp(data.Properties.VariableNames);  % Display the names of the variables in the dataset
unique(data.Label)  % Display unique values in the Label column

%% Step 2: Convert the 'Label' column into binary (Fault/No Fault)
binaryData = data;
binaryData.Label = double(binaryData.Label ~= 0);  % Convert non-zero labels to 1 (Fault), zero labels to 0 (No Fault)

%% Step 3: Separate features and labels
X = binaryData{:, 1:end-1};  % Features (all columns except the last)
X = normalize(X);            % Normalize features

Y = binaryData.Label;        % Labels (last column)

% Fix categorical encoding issue
Y = categorical(Y, [0, 1], {'No Fault', 'Fault'});  % Ensure correct class mapping

tabulate(Y)  % Display class distribution

%% One-hot encode the labels for patternnet
Y_onehot = onehotencode(Y, 2)';  % Convert to one-hot encoding (transpose to match input format)

%% Step 4: Split Data (80% Train, 20% Test)
cv = cvpartition(size(X,1), 'HoldOut', 0.2);  % Create a cross-validation partition for 80% train, 20% test
trainIdx = training(cv);
testIdx = test(cv);

X_train = X(trainIdx, :)';  % Transpose to match patternnet input format
Y_train = Y_onehot(:, trainIdx);
X_test = X(testIdx, :)';
Y_test = Y_onehot(:, testIdx);

%% Step 5: Define and Train Neural Network
% Experiment with Hyperparameters
net = patternnet([20, 10, 5]);  % Adjusted network architecture (two hidden layers with 20 and 10 neurons)
net.trainParam.epochs = 500;    % Increase epochs for better training
net.trainParam.lr = 0.0001;      % Learning rate
net.trainParam.showWindow = false;  % Disable training GUI to avoid interruptions

% Train the Neural Network
[net, tr] = train(net, X_train, Y_train);  % Train the network

%% Step 6: Predict on Test Data
Y_pred = net(X_test);  % Make predictions on the test data

% Convert outputs to class labels
[~, Y_pred_class] = max(Y_pred, [], 1);  % Get the predicted class labels
[~, Y_true_class] = max(Y_test, [], 1);  % Get the true class labels

% Convert 1,2 back to 0,1 for correct confusion matrix display
Y_pred_class = Y_pred_class - 1;  
Y_true_class = Y_true_class - 1;

%% Step 7: Evaluate the Model Performance

% Confusion Matrix
figure;
confusionchart(Y_true_class, Y_pred_class);
title('Confusion Matrix for Final Model');

%% Step 8: ROC Curve and AUC
% Extract probabilities for class 1 (Fault)
class1_probs = Y_pred(2, :);  % Probabilities of the positive class (Fault)

[Xroc, Yroc, ~, AUC] = perfcurve(Y_true_class, class1_probs, 1);  % Compute ROC curve and AUC

% Plot ROC Curve
figure;
plot(Xroc, Yroc);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve for Final Model (AUC = ' num2str(AUC) ')']);
grid on;

