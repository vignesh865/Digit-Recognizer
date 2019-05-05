function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================
J=cost(theta, X, y,m,lambda);

equation = X*theta ;
g_equation=sigmoid(equation);
error = (g_equation-y);

for feature_index=1:size(X)(2)

  error_f =error.*X(:,feature_index);
  grad(feature_index) = sum(error_f)/m;
  
end

grad(2:size(grad)) = grad(2:size(grad)) + ((lambda/m)*(theta(2:size(grad))));

end

function J=cost(theta,X,y,m,lambda)
%Consider hypothesis equation as y=theta0+(theta1*x1)+(theta2*x2)
equation = X*theta ;
g_equation_1=sigmoid(equation);
log_g_1=log(g_equation_1);
term1=y.*log_g_1;


g_equation_0=1-sigmoid(equation);
log_g_0=log(g_equation_0);
term0=(1-y).*log_g_0;

cost_function=sum(-1.*(term1+term0));
J=cost_function/m ;


theta_dash=theta(2:size(theta)).^2;
regularization_term = (sum(theta_dash) * lambda)/(2*m);
J=J+regularization_term;


endfunction

