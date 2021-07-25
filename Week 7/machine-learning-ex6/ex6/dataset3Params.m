function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% SOLUTION

result = ones(64,3);
errow=1;

for c = [0.01 0.03 0.1 0.3 1, 3, 10 30],
    for sig = [0.01 0.03 0.1 0.3 1, 3, 10 30],
        [model] = svmTrain(X,y,c, @(x1, x2) gaussianKernel(x1, x2, sig));
        predictions = svmPredict(model, Xval);
        pred_er = mean(double(predictions ~= yval));
        result(errow,:) = [c,sig,pred_er];
        errow = errow+1;
    end;
end;


[val ind]=min(result(:,3));
C = result(ind,1);
sigma = result(ind,2);
  
end
