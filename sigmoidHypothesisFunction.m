function y = sigmoidHypothesisFunction(z)
%This  function returns the sigmoid  value for the  z passed to it.
    y = 1.0 ./ (1+exp(-z));
end