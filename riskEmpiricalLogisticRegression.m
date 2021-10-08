function a = riskEmpiricalLogisticRegression(x,y,theta)
%This function calculates the empirical risk for the logistic regression
%model
    a = mean((y-1).*log(1-sigmoidHypothesisFunction(x*theta'))- ...
        y.*log(sigmoidHypothesisFunction(x*theta')));
end