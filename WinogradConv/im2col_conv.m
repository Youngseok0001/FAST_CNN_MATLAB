function [ convolved_output ] = im2col_conv( images, filters, conv )

images_dim = size(images,1);
images_depth = size(images,3);
images_num = size(images,4);
filters_dim =  size(filters,1);
output_dim = images_dim - filters_dim + 2*conv.padding +1;

images_padded = padarray(images, [conv.padding conv.padding]);
image_patches_temp = im2col_4D_sliding_v1(images_padded,[filters_dim filters_dim],[1 1]);
image_patches = permute(image_patches_temp,[1 3 2 4]);

images_im2col = reshape(image_patches,filters_dim * filters_dim * images_depth, []);
filters_im2col = reshape(filters, filters_dim * filters_dim * images_depth, [] );

convolved_output = permute(reshape(images_im2col' * filters_im2col,...
                   output_dim, output_dim, images_num,[]),[1 2 4 3]);
            
end
