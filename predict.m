function p = predict(Theta1, Theta2, X)
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
%theta matrix has extra column added so need to make sure extra is added
%here so the matrix multiplication works


%need to calculate the values
%hypothesis = sigmoid (theta * x)
%use sigmoid function to get values for hypothesis
%then use max to find what the best prediction is for our values
%we don't actually need the probabilities; just the index
%aka which class has the max probability?
%then output the predictions

%z_j = theta_j-1 * a_j-1
%a_j = sigmoid(z_j)
%z_j+1 = theta_j * a_j
%etc


%tranpose bc need matrix dimensions to match
z2 = Theta1 * X'; 
a2 = sigmoid(z2); 

a2 = a2';
ma2 = size(a2, 1);
a2 = [ones(ma2, 1) a2];
%add column of ones to a2
%theta matrix has extra column added so need to make sure extra is added
%here so the matrix multiplication works

z3 = Theta2 * a2'; 
hypothesis = sigmoid(z3); 

[probval, index] = max(hypothesis); 
p = index'; 


% =========================================================================


end
