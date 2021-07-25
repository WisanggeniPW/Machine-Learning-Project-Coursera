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

a1 = [ones(m, 1) X];
z2 = Theta1 * a1';
a2 = sigmoid(z2);
a2 = [ones(1, m); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

yVec = zeros(num_labels, m);
for i = 1 : m
  yVec(y(i), i) = 1;
end;

for i = 1 : m
  J += sum(-1 * yVec(:, i) .* log(a3(:, i)) - (1 - yVec(:, i)) .* log(1 - a3(:, i)));
end; 
J = J / m;

J = J + lambda * (sum(sum(Theta1(:, 2 : end) .^ 2)) + sum(sum(Theta2(:, 2 : end) .^ 2))) / 2 / m;

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for i = 1 : m
  delta3 = a3(:, i) - yVec(:, i)
  delta2 = (Theta2' * delta3)(2 : end, :) .* sigmoidGradient(z2(:, i));
  Delta2 += delta3 * a2(:, i)';
  Delta1 += delta2 * a1(i, :);
end;

Theta2_grad = Delta2 / m;
Theta1_grad = Delta1 / m;

Theta2_grad(:, 2 : end) = Theta2_grad(:, 2 : end) .+ lambda * Theta2(:, 2 : end) / m;
Theta1_grad(:, 2 : end) = Theta1_grad(:, 2 : end) .+ lambda * Theta1(:, 2 : end) / m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
