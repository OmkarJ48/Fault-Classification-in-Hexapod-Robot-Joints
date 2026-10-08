data = readtable('Hexapod_One_Joint_data_set.csv', 'VariableNamingRule', 'preserve');
disp(data.Properties.VariableNames);
unique(data.Label)
% converting data into binary and saving 
class(data.Label)  
binaryData = data;
binaryData.Label = double(binaryData.Label ~= 0); 
writetable(binaryData, 'binary_dataset.csv');
summary(binaryData);

% Load the binary dataset
data = readtable('binary_dataset.csv', 'VariableNamingRule', 'preserve');

% Separate features and labels
X = data{:, 1:end-1};  % Features (all columns except the last)
X = normalize(X);
Y = data.Label;        % Labels (last column)

% Convert labels to categorical for pattern recognition
Y = categorical(Y);  % Convert to categorical format
tabulate(Y)  % Shows how many samples per class

% Convert labels to one-hot encoding for patternnet
Y_onehot = onehotencode(Y, 2)';  % Convert to one-hot encoding (transpose to match input format)

% Split the dataset into training (80%) and test (20%)
cv = cvpartition(size(X,1), 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

X_train = X(trainIdx, :)';  % Transpose to match the input format for patternnet
Y_train = Y_onehot(:, trainIdx);
X_test = X(testIdx, :)';
Y_test = Y_onehot(:, testIdx);


% Define Neural Network
net = patternnet([10 5 1]);  % Create the network with 20 neurons in the first hidden layer and 10 in the second layer


% Set the epoch value
net.trainParam.epochs = 300;  % Set epochs to 100 for training

net.trainParam.lr = 0.0001;  % Learning rate

% Train the Neural Network
[net, tr] = train(net, X_train, Y_train);

% View the network
view(net);

% Predict on Test Data
Y_pred = net(X_test);

% Convert outputs to class labels
[~, Y_pred_class] = max(Y_pred, [], 1);
[~, Y_true_class] = max(Y_test, [], 1);

% Confusion Matrix
figure;
confusionchart(Y_true_class, Y_pred_class);
title('Confusion Matrix');

% Compute ROC Curve
[Xroc, Yroc, ~, AUC] = perfcurve(Y_true_class, Y_pred(2, :), 1);

% Plot ROC Curve
figure;
plot(Xroc, Yroc);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(AUC) ')']);
grid on;

trainPerf = tr.best_perf;
valPerf = tr.best_vperf;
testPerf = tr.best_tperf;

fprintf('Training Performance: %f\n', trainPerf);
fprintf('Validation Performance: %f\n', valPerf);
fprintf('Test Performance: %f\n', testPerf);

