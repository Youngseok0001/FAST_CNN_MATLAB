function [train_images,train_labels, test_images, test_labels] = Load_SVHN_data( )


train_data = load('train_32x32.mat');
test_data = load('test_32x32.mat');

train_image = double(train_data.X);  
test_image = double(test_data.X);  
          
train_image =  double(reshape(train_image,32,32,3,[])/255);
test_image =  double(reshape(test_image,32,32,3,[])/255);

%Mean normalisation
data_mean = mean(train_image, 4);
train_images = bsxfun(@minus, train_image, data_mean);
test_images = bsxfun(@minus, test_image, data_mean);

train_labels = double([train_data.y]);
test_labels = double([test_data.y]);

          

end

