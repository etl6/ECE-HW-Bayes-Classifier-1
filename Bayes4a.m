%% Bayes Classifier 
%% Emma LaPorte
%% HW4 - Problem 4
% BAYES CLASSIFIER
clear all; close all; clc;

%% Create dataset
% data set 1
N=200;
% mu1=[1 1];
% sigma1=[2 1;1 3];
% Features_1 = mvnrnd(mu1,sigma1, N);
truth_1 = zeros(N,1);
% % data set 2
% mu2 = [5 7];
% sigma2 = [5 -1;-1 4];
% Features_2 = mvnrnd(mu2,sigma2,N);
truth_2 = ones(N,1);

load('Features.mat') % produces Features_1 and Features_2 same for consistency

% plotting the data
figure()
plot(Features_1(:,1),Features_1(:,2),'c*')
hold on
plot(Features_2(:,1),Features_2(:,2),'ro')
legend('Class 0','Class 1')
title('Two distinct classes Problem 3')
xlabel('Feature 1'); ylabel('Feature 2')

% setting up and reshaping features and truth labels
%xTrain = [Features_1 Features_2];
truth  = [truth_1 truth_2];
truth  = reshape(truth,[2*N,1]);
xTrain(:,1) = [Features_1(:,1);Features_1(:,2)];
xTrain(:,2) = [Features_2(:,1);Features_2(:,2)];

% to plot decision stat surface
x1Range = max(xTrain(:,1))- min(xTrain(:,1));
x2Range = max(xTrain(:,2))- min(xTrain(:,2));
x1 = linspace(min(xTrain(:,1))-0.2*x1Range,max(xTrain(:,1))+0.2*x1Range,251);
x2 = linspace(min(xTrain(:,2))-0.2*x2Range,max(xTrain(:,2))+0.2*x2Range,251);
% Create the grid of test data points
[xTest1,xTest2] = meshgrid(x1,x2);
% Each column is a feature
xTest = [xTest1(:) xTest2(:)];

len = length(xTest(:,1));
thr=1; % thr=beta
decStat_H0 = zeros(1,len);
decStat_H1 = zeros(1,len);
dsTest = zeros(1,len);

%% applying Bayes classifier

mean1 = mean(Features_1);
mean2 = mean(Features_2);

% 4a) covariances
% cov1  = cov(Features_1);
% cov2  = cov(Features_2);

% 4c)  
% cov1  = cov(Features_1);
% cov1(1,2) = 0;
% cov1(2,1) = 0;
% cov2  = cov(Features_2);
% cov2(1,2) = 0;
% cov2(2,1) = 0;
% % 

% % 4e) 
% Features_demean1 = Features_1 - mean1;
% Features_demean2 = Features_2 - mean2;
% Features_concat  = [Features_demean1;Features_demean2];
% cov1  = cov(Features_concat);
% cov2  = cov1;
% 
% % 
% % % 4g) 
Features_demean1 = Features_1 - mean1;
Features_demean2 = Features_2 - mean2;
Features_concat  = [Features_demean1;Features_demean2];
cov1  = cov(Features_concat);
cov1(1,2)=0;
cov1(2,1)=0;
cov2  = cov1;

% calculate each decStat 
for i=1:len 
    decStat_H0(i)=mvnpdf(xTest(i,:), mean1, cov1);
    decStat_H1(i)=mvnpdf(xTest(i,:), mean2, cov2);
    dsTest(i)=decStat_H0(i)/decStat_H1(i); % pg 7 slides
end

% classification rule, compare to 0.5/0.5 = 1
for j=1:len 
    if dsTest(j) > thr
        dsTest(j) = 1; % decide H1
    else 
        dsTest(j) = 0; % decide H0
    end
end

% Image the decision statistic surface
% reshape dsTest for imagesc
dsTest = reshape(dsTest,length(x2),length(x1));
figure()
imagesc(x1([1 end]),x2([1 end]),dsTest)
colorbar
set (gca,'YDir','normal')
% Add the training data points to the surface
hold on
% H0
plot(Features_1(:,1),Features_1(:,2),'c*')
% H1
plot(Features_2(:,1),Features_2(:,2),'ro')
contour(xTest1, xTest2, dsTest,'Linewidth', 2, 'color', 'k')
% plot tidy up
title('Bayes Classifier')
xlabel('Feature 1'); ylabel('Feature 2');
legend('H0','H1')
hold off
