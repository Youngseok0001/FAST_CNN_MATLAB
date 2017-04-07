function [params] = stack2params(stack)

% Converts a "stack" structure into a flattened parameter vector and also
% stores the network configuration. This is useful when working with
% optimization toolboxes such as minFunc.
%
% [params, netconfig] = stack2params(stack)
%
% stack - the stack structure, where stack{1}.w = weights of first layer
%                                    stack{1}.b = weights of first layer
%                                    stack{2}.w = weights of second layer
%                                    stack{2}.b = weights of second layer
%                                    ... etc.

params = [];

    
for d = 1:numel(stack)
  
    params = [params ; stack{d}.W(:) ; stack{d}.b(:) ];

     end
end
