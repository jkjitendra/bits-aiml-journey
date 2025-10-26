%PCA example in the slide deck
clear all; %clears the workspace
close all; %clears the figure windows
%The data matrix is stored in A
A = [1.11 10;1.21 12;1.36 13;1.49 15;1.63 16;1.68 17;1.83 18; 1.88 19;1.95 20];

plot(A(:,1),A(:,2),'r*');
legend("raw data");
axis square;

%Store the mean in mu
mu = mean(A);

%Normalize the data by subtracting the mean
A_normalized = A - mu;

%Construct the covariance matrix X
X = 1/9*A_normalized'*A_normalized;

%Compute the eigenvalues and eigenvectors of X;
[E V] = eigs(X);

%The first eigenvector corresponding to the largest eigenvalue is the first principal component and is stored in b1;
b1 = E(:,1);

%X_tilde is the compressed version of A
X_tilde = b1'*A_normalized';

%Plot the values along the principal component
X_pc=b1*X_tilde;
figure;
plot(X_pc(1,:),X_pc(2,:),'b*');
legend("Along principal component");
axis square;

%Mean adjusted principal components
X_pc_ma = X_pc + mu';
figure;
plot(X_pc_ma(1,:),X_pc_ma(2,:),'k*');
legend("Mean adjusted principal component");
axis square;


