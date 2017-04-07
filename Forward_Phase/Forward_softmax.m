function [ activations , activations_pooled] = Forward_softmax(stack, activationsPooled_prev )

activations = stack.W * activationsPooled_prev + repmat(stack.b,...
               [1,size(activationsPooled_prev, 2)]);
activations = exp(activations);
activations = bsxfun(@rdivide, activations, sum(activations,1)); 
activations_pooled = activations;
end

