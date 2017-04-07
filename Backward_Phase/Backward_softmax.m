function [ grad_W ,grad_b, delta ] = Backward_softmax( activations,...
                                      activations_prev, labels)

indicator = zeros(size(activations) );
index = sub2ind(size(activations),labels',1:size(activations,2));
indicator(index) = 1;
delta = -(indicator - activations);
grad_W = delta * (activations_prev)';
grad_b = sum(delta,2);
    
end

