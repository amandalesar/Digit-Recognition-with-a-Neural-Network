function g = sigmoidGradient(z)
% sigmoid = 1.0 ./ (1.0 + exp(-z));
g = sigmoid(z) .* (1-sigmoid(z)); 
end
