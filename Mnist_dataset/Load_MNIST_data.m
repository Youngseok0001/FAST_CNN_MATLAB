function [ images, labels, testImages, testLabels ] = Load_MNIST_data(  )

% Load MNIST Train
images = loadMNISTImages('../Mnist_dataset/train-images-idx3-ubyte');
images = reshape(images, 28, 28, 1, []);
images = imresize(images,[32 32]);
labels = loadMNISTLabels('../Mnist_dataset/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

% Load MNIST Test
testImages = loadMNISTImages('../Mnist_dataset/t10k-images-idx3-ubyte');
testImages = reshape(testImages,28,28,1,[]);
testImages = imresize(testImages,[32 32]);
testLabels = loadMNISTLabels('../Mnist_dataset/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10



% Mean normalisation

data_mean = mean(images, 4);
images = bsxfun(@minus, images, data_mean);
testImages = bsxfun(@minus, testImages, data_mean);


end

