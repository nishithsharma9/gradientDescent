function [optimumTheta] = linearLogisticRegression(x,y,tolerance,stepValue)%,maxIteration)
% This function is used to find the optimum theta using gradient descent
% for the linear logistic regression model. 
% This can be used ove the data set of dynamic dimension
% Input will be the tolerance value, also called eta
% Input will also be stepValue, or the alpha or the gradient factor
% Usage
% optimumTheta = (x,y,1e-4,0.1)
% Where X is the n dimensional input, and y is the binary outputs for the
% classigication model.

% Adding a column of ones in starting of the data, This is needed as this
% will correspond to theta0
x = [(ones(length(x),1)),x];

% Assuming theta to be ones initially
theta_old = ones(1,width(x));

% For the first theta_new we perform update using the step value gradient
% descent
theta_new = theta_old-stepValue*(sigmoidHypothesisFunction(x*theta_old')-y)'*x; 

iteration = 1;

% The error and wrong prediction percentage will be calculated and stored
% in the following vectors. These two will be the 2 cross validations
errEmpiricalVector = [];
wrongPredictionPercentageVector = [];

while( (norm(theta_new-theta_old)>= tolerance) )
    %updating the theta_values
    theta_old = theta_new;
    theta_new = theta_new-stepValue*(sigmoidHypothesisFunction(x*theta_new')-y)'*x;    
    iteration = iteration+1;
    
    % There will be two ways to validate.
    % First, Minimize the Empirical risk
    % Second, Minimize wrong predictions
    currentPredictions=sigmoidHypothesisFunction(x*theta_new');
    currentPredictions(currentPredictions>0.5)=1;
    currentPredictions(currentPredictions<=0.5)=0;
    wrongPrediction=xor(currentPredictions,y);
    wrongPredictionPercentage= sum(wrongPrediction)*100/length(y);
    wrongPredictionPercentageVector = [wrongPredictionPercentageVector; wrongPredictionPercentage];
    errEmpiricalVector = [errEmpiricalVector; riskEmpiricalLogisticRegression(x,y,theta_new)];
end

%Displays the optimum theta, and the  number of iterations
optimumTheta = theta_new;
iteration = iteration;
length(errEmpiricalVector);
diffTheta = norm(theta_new-theta_old)

iteratorVector=(1:iteration-1);
iteratorVector2=(1:length(wrongPredictionPercentageVector));

%plotting the cross validation graph using the gradient descent and also
%the percentage correct predictions
figure("Name","Empirical Risk Over Iterations of Gradient Descent")
plot(iteratorVector,errEmpiricalVector,'red')
xlabel('Iterations in Gradient Descent')
ylabel('Risk Empirical')
figure("Name","Wrong Prediction Percentage Over Iterations of Gradient Descent")
plot(iteratorVector2,wrongPredictionPercentageVector,'green')
xlabel('Iterations in Gradient Descent')
ylabel('Percentage miscalculation')