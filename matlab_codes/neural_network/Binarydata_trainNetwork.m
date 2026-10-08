% Load the dataset
data = readtable('Hexapod_One_Joint_data_set.csv', 'VariableNamingRule', 'preserve');
disp(data.Properties.VariableNames);
unique(data.Label)

% Convert data to binary
binaryData = data;
binaryData.Label = double(binaryData.Label ~= 0);  % Convert to binary (0 = No Fault, 1 = Fault)
writetable(binaryData, 'binary_dataset.csv');
summary(binaryData);

% Load the binary dataset
data = readtable('binary_dataset.csv', 'VariableNamingRule', 'preserve');

% Separate features and labels
X = data{:, 1:end-1};  % Features (all columns except the last)
X = normalize(X);  % Normalize the features
Y = data.Label;    % Labels (last column)

% Convert labels to categorical for classification tasks
Y = categorical(Y);  % Convert labels to categorical format
tabulate(Y)  % Shows how many samples per class

% Split the dataset into training (70%) and test (30%)
cv = cvpartition(size(X,1), 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

X_train = X(trainIdx, :);  % Training features
Y_train = Y(trainIdx);     % Training labels
X_test = X(testIdx, :);    % Test features
Y_test = Y(testIdx);       % Test labels

% Define the layers for the feedforward neural network
layers = [
    featureInputLayer(size(X, 2))  % Input layer (size equal to the number of features)
    fullyConnectedLayer(10)        % First hidden layer with 10 neurons
    reluLayer                      % ReLU activation function
    fullyConnectedLayer(5)         % Second hidden layer with 5 neurons
    reluLayer                      % ReLU activation function
    fullyConnectedLayer(2)         % Output layer (binary classification, 2 classes)
    softmaxLayer                   % Softmax layer for probabilities
    classificationLayer            % Classification layer
];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 300,          ... % Set the number of epochs
    'MiniBatchSize', 128,      ... % Set batch size (adjust based on your data)
    'InitialLearnRate', 0.0001, ... % Learning rate
    'Shuffle', 'every-epoch',  ... % Shuffle data every epoch
    'Verbose', true,           ... % Show training progress
    'Plots', 'training-progress');  % Plot training progress

% Train the network using trainNetwork
net = trainNetwork(X_train, Y_train, layers, options);

% Predict on test data
Y_pred = classify(net, X_test);

% Confusion Matrix
figure;
confusionchart(Y_test, Y_pred);
title('Confusion Matrix');

% Get predicted probabilities for class 1 (fault)
scores = predict(net, X_test); % Corrected to return only one output

% Compute ROC Curve
[Xroc, Yroc, ~, AUC] = perfcurve(double(Y_test), scores(:,2), 1);

% Plot ROC Curve
figure;
plot(Xroc, Yroc, 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(AUC) ')']);
grid on;