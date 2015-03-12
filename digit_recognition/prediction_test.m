
% Load Training Data
fprintf('Loading and Visualizing Data ...\n\n')

load('train_data.mat');
m = size(X, 1);

params=load('weights.mat'); % contains the weights of the neural net

Q = randperm(m);

%%%% NN FOR A GIVEN IMAGE

%I=imread('5.png');
%J=rgb2gray(I);
%K=imcomplement(J); %the digit has to be in white color and the background has to be in black
%L=imresize(K,[28 28]);
%O=reshape(L,1,28*28);
%Q = double(O) ./ 255 ;


Theta1 = params.Theta1; 
Theta2 = params.Theta2;

for i = 1:m

fprintf('\nDisplaying Selected Image....\n');
shownumber(X(Q(i), :)); 
pred = predict(Theta1,Theta2,X(Q(i),:))
fprintf('\nMachine Prediction: %d (actual digit %d)\n', pred, mod(pred, 10));
 % Pause
fprintf('Program paused. Press enter to continue.\n');
pause
end
