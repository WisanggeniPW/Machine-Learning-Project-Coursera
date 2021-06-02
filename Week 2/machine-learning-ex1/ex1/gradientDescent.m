function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    i=1:m;
    theta(1)=theta(1)-((alpha*(1/m))*(sum(((theta(1) + theta(2) .* X(i,2)) - y(i)))));
    theta(2)=theta(2)-((alpha*(1/m))*(sum(((theta(1) + theta(2) .* X(i,2)) - y(i)) .* X(i,2))));

    J_history(iter) = computeCost(X, y, theta);

end

end
