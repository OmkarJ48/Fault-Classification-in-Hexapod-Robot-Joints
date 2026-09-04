%% 1️⃣ Load & Preprocess Data
data = readtable('Hexapod_One_Joint_data_set.csv', 'VariableNamingRule', 'preserve');

% Convert labels to binary (0 = Class 0, 1 = Class 1)
data.Label = double(data.Label ~= 0);

% Separate classes
minority = data(data.Label == 0, :);  % Class 0
majority = data(data.Label == 1, :);  % Class 1

% Ensure equal number of samples
num_to_keep = height(minority);
majority_subset = majority(randperm(height(majority), num_to_keep), :); % Randomly select equal samples

% Combine into a balanced dataset
balanced_data = [minority; majority_subset];

% Shuffle the dataset
balanced_data = balanced_data(randperm(height(balanced_data)), :);

% Save the balanced dataset
writetable(balanced_data, 'equal_balanced_dataset.csv');

%% 2️⃣ Load the Balanced Dataset
data = readtable('equal_balanced_dataset.csv', 'VariableNamingRule', 'preserve');

% Separate features and labels
X = data{:, 1:end-1};  % Features (all columns except the last)
X = normalize(X);      % Normalize features
Y = data.Label;        % Labels (0 or 1)

% Check class balance
tabulate(Y)

%% 3️⃣ Train-Test Split (80% Training, 20% Testing)
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
trainIdx = training(cv);
testIdx = test(cv);

X_train = X(trainIdx, :)';  % Transpose for `patternnet`
Y_train = double(Y(trainIdx))';  % Binary labels (0 or 1)
X_test = X(testIdx, :)';
Y_test = double(Y(testIdx))';

%% 4️⃣ Define & Train Neural Network
net = patternnet([10 5]);  % Two hidden layers with 10 and 5 neurons

% Set training parameters
net.trainParam.epochs = 300;  % Number of training iterations
net.trainParam.lr = 0.001;    % Learning rate for better convergence
net.layers{end}.transferFcn = 'logsig';  % Sigmoid activation for binary classification

% Train the Neural Network
[net, tr] = train(net, X_train, Y_train);

% View the network
view(net);

%% 5️⃣ Make Predictions & Evaluate Performance
Y_pred = net(X_test);  % Get prediction probabilities
Y_pred_class = double(Y_pred > 0.5);  % Convert probabilities to binary labels

% Confusion Matrix
figure;
confusionchart(Y_test, Y_pred_class);
title('Confusion Matrix');

% Compute ROC Curve & AUC Score
[Xroc, Yroc, ~, AUC] = perfcurve(Y_test, Y_pred, 1);

% Plot ROC Curve
figure;
plot(Xroc, Yroc, 'b', 'LineWidth', 2);
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