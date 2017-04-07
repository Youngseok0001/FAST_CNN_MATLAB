function [ activations, activations_pooled] = Forward_fc(stack, activationsPooled_prev )

global Activation_function;


activations = stack.W * activationsPooled_prev + repmat(stack.b,...
               [1,size(activationsPooled_prev, 2)]);
          %mask = 2 * round(rand(size(activations)));
           %activations = sigmoid(activations);
           
if strcmp(Activation_function,'RELU')
          
activations = relu(activations); 
           
elseif strcmp(Activation_function,'SIGMOID')

activations = sigmoid(activations); 
                       
end

activations_pooled = activations;

end


