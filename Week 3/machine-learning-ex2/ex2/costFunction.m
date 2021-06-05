function [J, grad] = costFunction(theta, X, y)

m = length(y); 

J = 0;
grad = zeros(size(theta));
hx = sigmoid(X*theta);
J = (1/m)*sum((-y.*log(hx))-((1-y).*log(1-hx))); % scalar
grad = (1/m)* ((X'*(hx-y)));


end
