%problem 4
clear
load('dataset.mat')

%Assuming the below two values.
%For high step values we might miss the local minima
%For really small tolerance  the code will  execute and crash due to the
%limited computation power. If we take higher tolerance, then theta will
%prematurely stop. I wanted to take tolerance as 1e-6, but my laptop got
%stuck, thus for simplicity assumed it to be 1e-4
stepValue = 0.1
tolerance = 1e-4

optimumTheta = linearLogisticRegression(X,Y,tolerance,stepValue)

%for this specific problem, as we are given the 3 input features, we can
%plot it in a 3D plane. If the dimension inreases, the model above  will
%work fine, just that we will not be able to plot it.
figure("Name","3D plane for data and  model plot")
for i = 1:length(X)
    if (Y(i)==1)
        plot3(X(i,1),X(i,2),X(i,3),'o','MarkerFaceColor','red')
        hold on
    else
        plot3(X(i,1),X(i,2),X(i,3),'o','MarkerFaceColor','green')
        hold on
    end
end

% Plotting the surface for the model predicted
[p1,p2] = meshgrid(0:0.01:1,-1.5:0.01:1.1);
p3 = (-1/optimumTheta(4))*(optimumTheta(2)*p1 + optimumTheta(3)*p2 + optimumTheta(1));
surf(p1,p2,p3)
shading interp;
xlabel('Feature1')
ylabel('Feature2')
zlabel('Feature3')
legend('0 output','1 output','Model Surface')
