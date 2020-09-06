function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%J: remember hypothesis = final a
%z_j = theta_j-1 * a_j-1
%a_j = sigmoid(z_j)
%z_j+1 = theta_j * a_j
%etc
%add column of ones to a2
%theta matrix has extra column added so need to make sure extra is added
%here so the matrix multiplication works

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2); 
ma2 = size(a2, 1);
a2 = [ones(ma2, 1) a2];
z3 = a2 * Theta2'; 
a3 = sigmoid(z3); 
hypothesis = a3; 
%recall a3 = hypothesis = g(z3)

%need to reformat y values as vectors containing 0 or 1
%so instead of y having 1,2,3,4,4,5,6,7,8,9,10
%y now has rows which have 1 identifying number and 0 otherwise
%ex: y=1 --> y = [1 0 0 0 0 0 0 0 0 0]
%ex: y=4 --> y = [0 0 0 1 0 0 0 0 0 0]
numvector = [1:num_labels];
% for i = 1:length(y)
%     ynn(i,:) = (y(i) == numvector);
% end
ynn = (y==numvector);

J = (-1/m) * sum(sum(ynn.*log(hypothesis) + (1-ynn).*log(1-hypothesis))); 
%2 sums in J now

%add regularization to cost function
% temptheta1 = Theta1;
% temptheta1(1) = 0; 
% temptheta2 = Theta2;
% temptheta2(1) = 0; 

temptheta1 =  Theta1(:,2:end);
temptheta2 =  Theta2(:,2:end);
%removes Theta_0 element from each matrix

sum1 = sum(sum(temptheta1.^2));
sum2 = sum(sum(temptheta2.^2));
regterms = (lambda/(2*m))* (sum1 + sum2); 
J = J + regterms; 

%%%%%%%%%%%%%%%%%%%%%%%%
% Backpropagation for gradients

del1 = zeros(size(Theta1));
del2 = zeros(size(Theta2));
for t = 1:m
%already calculated a1, a2, a3 above
%so we can directly pull out the tth training example
    a1t = a1(t,:);
    a2t = a2(t,:);
    a3t = a3(t,:); % = hypothesis
    yt = ynn(t,:); %actual value
%     a1 = [ones(m,1) X];
%     z2 = a1 * Theta1';
%     a2 = sigmoid(z2); 
%     ma2 = size(a2, 1);
%     a2 = [ones(ma2, 1) a2];
%     z3 = a2 * Theta2'; 
%     a3 = sigmoid(z3); 
%     hypothesis = a3; 
    delta3 = a3t - yt;
    delta2 = (delta3*Theta2)'.*sigmoidGradient([1 a1t * Theta1']');
    del1 = del1 + delta2(2:end)*a1t;
    del2 = del2 + delta3'*a2t;
end
%keep checking matrix dimensions
%that's where you're running into issues!!!

Theta1_grad = (1/m)*del1; 
Theta2_grad = (1/m)*del2; 

%add regularization terms to gradient
%gradmat = gradmat + lambda/m*parametermat j>=1
%gradmat = gradmat j=0
temptheta1 =  Theta1(:,2:end);
temptheta2 =  Theta2(:,2:end);
%removes Theta_0 element from each matrix
%add in a column of zeros
%need column of zeros becuse need zero when j = 0
%and j denotes the columns
temptheta1 = [zeros(size(Theta1,1),1) temptheta1];
temptheta2 = [zeros(size(Theta2,1),1) temptheta2];

Theta1_grad = Theta1_grad + (lambda/m)*temptheta1;
Theta2_grad = Theta2_grad + (lambda/m)*temptheta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
