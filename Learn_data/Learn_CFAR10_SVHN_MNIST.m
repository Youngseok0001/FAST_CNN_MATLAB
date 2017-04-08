%% My CNN
clear;
clc;

addpath ../Cfar10_dataset/;
addpath ../SVHN_dataset/;
addpath ../Mnist_dataset/;
addpath ../Backward_Phase/;
addpath ../Forward_Phase/;
addpath ../InitParams/;
addpath ../Relu_Sigmoid/;
addpath ../WinogradConv/;
addpath ../Im2colConv/;
addpath ../Optimization_method/;
%%======================================================================
%% STEP 0: Initialize Parameters and Load Data
global imageDim;
global numClasses;
global input_size;
global Convolution_type; 
global Data_type;
global Activation_function;
global Optimization_method;
global Pooling;

% Initialize network

Data_type = 'MNIST'; % OR [CIFAR10, SVHN]
Convolution_type = 'WINOGRAD'; % OR WINOGRAD
Activation_function = 'RELU';% OR SIGMOID
Optimization_method = 'ADAGRAD'; % OR SGD
Pooling = 'MEAN'; % OR [MAX,MIN] 

fprintf('Data_type = %s \n',Data_type);
fprintf('Convolution_type = %s\n',Convolution_type);
fprintf('Activation_function = %s\n',Activation_function);
fprintf('Optimization_method = %s\n',Optimization_method);
fprintf('Pooling_type = %s\n',Pooling);

% Load data
if strcmp(Data_type,'CIFAR10')

[images, labels, testImages, testLabels] = Load_CIFAR10_data( );

elseif strcmp(Data_type,'MNIST')
    

[images, labels, testImages, testLabels] = Load_MNIST_data( );

elseif strcmp(Data_type,'SVHN')

[images, labels, testImages, testLabels] = Load_SVHN_data( );

end

%layer specifications

imageDim = size(images,1);
input_size = size(images);
numClasses = 10;

[conv,full] = Cnn_spec_Mnist_1();


% Initialise parameters for convlayers
for convLayerNum = 1: numel(conv)
    
   [stack{convLayerNum}.W, stack{convLayerNum}.b, output_size] = ...
                   InitParams_conv(input_size, conv{convLayerNum});
   input_size = output_size; % output becomes the input of next layer               
end


for fullLayerNum = numel(conv) + 1 : numel(full)
   
    [stack{fullLayerNum}.W, stack{fullLayerNum}.b, output_size ] = ...
                         InitParams_fl(input_size, full{fullLayerNum});      
    input_size = output_size; % output becomes the input of next layer                           
end

% Convert stack to array
theta = stack2params(stack);


%Convert Input data, labels and parameters into single GpuArrays 
images = single(images);
labels = single(gpuArray(labels));
theta = single(gpuArray(theta));

%% STEP 1: Learn Parameters

options.epochs = 20;
options.minibatch = 256;
options.alpha = 1e-3;


if strcmp(Optimization_method,'ADAGRAD')

[opttheta, cost] = minFuncAdaGrad(@(x,y,z) CnnCost_dropout(x,y,z,numClasses,conv,...
                      full),theta,images,labels,options);
elseif strcmp(Optimization_method,'SGD')
    
[opttheta, cost] = minFuncSGD(@(x,y,z) CnnCost_dropout(x,y,z,numClasses,conv,...
                      full),theta,images,labels,options);
end
                  
save('optimized_theta.mat','opttheta');
save('optimized_cost.mat','cost');


%%======================================================================
%% STEP 2: Test

temp = load('optimized_theta.mat');
opttheta =  gpuArray(temp.opttheta);

for  i = 1 : options.minibatch : size(testLabels,1) - options.minibatch
    
[~, ~, preds(i:i+options.minibatch-1)]=CnnCost_dropout(opttheta,...
         testImages(:,:,:,i:i+options.minibatch-1),testLabels(i:i+options.minibatch-1),numClasses,...
        conv,full,true);
end          
                    
acc = sum(preds'==testLabels(1:i+options.minibatch-1))/length(preds(1:i+options.minibatch-1));
fprintf('Accuracy is %f\n',acc);
