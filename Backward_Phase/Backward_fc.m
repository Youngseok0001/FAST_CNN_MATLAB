function [ grad_W, grad_b, delta_new ] = Backward_fc( activations,...
                                          activations_prev, stack, delta)                                     
                                      
global Activation_function;
                                    
if strcmp(Activation_function,'SIGMOID')
    
delta_new = (stack.W' * delta) .* activations .*(1-activations);

elseif strcmp(Activation_function,'RELU')

delta_new = (stack.W' * delta) .* (activations >0);

end
grad_W = delta_new * activations_prev';
grad_b = sum(delta_new,2);   
     
end

