 clear all; close all; clc

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('train_data.mat');
m = size(X, 1);

% Randomly select 300 images from the data
sel = randperm(size(X, 1));
sel = sel(1:300);
shownumber(X(sel, :));


%Setup Parameters - NN layer sizes

input_layer_size  = 784 ;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;

% lambda - regularization factor
lambda_range = [3];
lambda_best = lambda_range(1);


for i=1:size(lambda_range,2)

   %Initialize NN Parameters for the 3-layer NN
   initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
   initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

   % Unroll parameters
   initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
   
   % Implement backprop and train network 
   fprintf('\nTraining Neural Network... \n\n')
   
   options = optimset('MaxIter', 400);

   lambda = lambda_range(i);

   costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    params=nn_params;
       
end

Theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nVisualizing Neural Network... \n')

pred = predict(Theta1, Theta2, X);
fprintf('\n Training Set Accuracy: %f\n', mean(double(pred  == y)) * 100);

save('weights1.mat');


