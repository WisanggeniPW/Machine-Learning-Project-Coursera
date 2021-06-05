function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);

J = 0;
grad = zeros(size(theta));
hx= sigmoid(X*theta)
  
term = (lambda/(2*m)) * sum(theta(2:end).^2);  
J = (1/m)*sum((-y.*log(hx))-((1-y).*log(1-hx))) + term;  
grad(1) = (1/m)* (X(:,1)'*(hx-y));                                 
grad(2:end) = (1/m)* (X(:,2:end)'*(hx-y))+(lambda/m)*theta(2:end);


end
