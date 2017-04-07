function [ grad_W, grad_b, delta_new ] = Backward_conv( activations, activationsPooled_prev,...
                                    stack, delta, conv, conv_temp)
      
global Convolution_type; 
global Activation_function;

if nargin == 5
    
    delta_temp = stack.W' * delta;  
    delta_temp = reshape(delta_temp,conv.outputDim,conv.outputDim,conv.numFilters,[]);
    
else
    
    if strcmp(Convolution_type,'WINOGRAD')
    
    delta_temp = wino_conv(delta, permute(rot90(stack.W, 2),[1 2 4 3]), conv_temp);
    
    elseif strcmp(Convolution_type,'IM2COL')

    delta_temp = im2col_conv(delta, permute(rot90(stack.W, 2),[1 2 4 3]), conv_temp);
    
    end

        
end
        
   delta_new = (1/conv.poolDim^2).* repelem(delta_temp,conv.poolDim,conv.poolDim,1,1);
        
   if strcmp(Activation_function,'SIGMOID')

        delta_new = delta_new .* activations .* (1-activations);
        
   elseif strcmp(Activation_function,'RELU')

        delta_new = delta_new .* (activations > 0 );
   end
        
        grad_W = im2col_conv(permute(activationsPooled_prev,[1 2 4 3]),...
                             permute(delta_new,[1 2 4 3]), conv);
        grad_W = permute(grad_W,[1 2 4 3]);

 temp = permute(delta_new,[1 2 4 3]);
 temp = sum(sum(sum(temp)));
 grad_b = reshape(temp,1,[]);
end


