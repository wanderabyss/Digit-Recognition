function g = sigmoidGradient(z)

% computes the gradient of the sigmoid function evaluated at z.
% This should work regardless if z is a matrix or a vector.

g = zeros(size(z));
g= sigmoid(z) .* (1-sigmoid(z));

end