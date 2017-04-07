function [cost] = Forward_cost( activations, labels)

cost_temp_log = log(activations);
index = sub2ind(size(activations),labels', 1:size(activations,2));
cost = -sum(cost_temp_log(index));
end

