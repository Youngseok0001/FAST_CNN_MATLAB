function [train_images,train_labels, test_images, test_labels] = Load_CIFAR10_data( )

train_data_1 = load('data_batch_1.mat');
train_data_2 = load('data_batch_2.mat');
train_data_3 = load('data_batch_3.mat');
train_data_4 = load('data_batch_4.mat');
train_data_5 = load('data_batch_5.mat');
test_data = load('test_batch.mat');

% Append image data
train_data = double([train_data_1.data;
              train_data_2.data;
              train_data_3.data;
              train_data_4.data;
              train_data_5.data]);  
          
% Augmentaion: horizontal flip and vertical flip          
train_images_nonflip =  double(permute(reshape(train_data',32,32,3,[]), [2 1 3 4]))/255;
train_images_flip_1 = flip(train_images_nonflip,1);
train_images_flip_2 = flip(train_images_nonflip,2);
train_images = cat(4,train_images_nonflip,train_images_flip_1, train_images_flip_2);
test_images =  double(permute(reshape(test_data.data',32,32,3,[]), [2 1 3 4]))/255;

%Mean normalisation
data_mean = mean(train_images, 4);
train_images = bsxfun(@minus, train_images, data_mean);
test_images = bsxfun(@minus, test_images, data_mean);

% Append Label data 
train_labels = double([train_data_1.labels;
              train_data_2.labels;
              train_data_3.labels;
              train_data_4.labels;
              train_data_5.labels]);
train_labels = [train_labels;train_labels;train_labels];           
train_labels(train_labels==0) = 10; % Remap 0 to 10
          
          
          
test_labels = double([test_data.labels]);
test_labels(test_labels==0) = 10; % Remap 0 to 10

          

end

