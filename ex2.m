%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = csvread('train.csv');

dataframe = data(2:end, :); 
X=dataframe(:, 2:end); 
y = dataframe(:, 1);

fprintf('Training data loaded')

test_data = csvread('test.csv');

test_dataframe = test_data(2:end, :); 


fprintf('Test data loaded')

m = size(X)(1);
n = size(X)(2);

lambda = 0.01;
num_labels = 10;
[all_theta] = oneVsAll(X, y,num_labels, lambda);


%% ================ Part 3: Predict for One-Vs-All ================



pred = predictOneVsAll(all_theta, test_dataframe)

%% ================ Part 3: Predict Accuracy ================

%pred = predictOneVsAll(all_theta, X)

%fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);