function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         

y_new = zeros(m,num_labels);
for i = 1:m
  y_new(i, y(i))=1;
end
  
% You need to return the following variables correctly 
J = 0;
  x_in = X;
  x_in = [ones(m,1) x_in];
  z2 = x_in*Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(m,1) a2];
  z3 = a2*Theta2';
  a3 = sigmoid(z3);
  hT = a3;
K = size(Theta2,1);
  
  
J = sum(sum(-y_new.*log(hT) - (ones(size(y_new))-y_new).*log(ones(size(hT))-hT)));
J = J/m;

sum_theta_2 = sum(sum((Theta1.*[zeros(size(Theta1,1), 1) Theta1(:,2:size(Theta1,2))]))) + sum(sum((Theta2.*[zeros(size(Theta2,1), 1) Theta2(:,2:size(Theta2,2))])));

J = J + lambda*0.5/m*(sum_theta_2);


Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

del_3 = hT-y_new; % mxL
Theta2_new = Theta2(:,2:end);
del_2 = (del_3*Theta2_new).*sigmoidGradient(z2); % mxh
D1 = del_2'*x_in; % mxh, mxf (features=400 here) => hxf
D2 = del_3'*a2; % mxL, mxh => Lxh





Theta1_add = [zeros(size(Theta1, 1),1) Theta1(:,2:end)];
Theta2_add = [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

Theta1_grad = D1/m + lambda/m*Theta1_add;
Theta2_grad = D2/m + lambda/m*Theta2_add;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

