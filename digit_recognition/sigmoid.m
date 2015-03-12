function [ g ] = sigmoid( z )

% Compute sigmoid functoon

g = 1 ./ (1+exp(-z));

end

