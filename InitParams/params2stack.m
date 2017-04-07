function stack = params2stack(params,input_size, conv, full)


% params - flattened parameter vector

%conv:
%conv.filterDim  -    Filter size for conv layer
%conv.filterDepth -   Depth of a filter
%conv.numFilters -    Number of filters for conv layer
%conv.poolDim    -    Pooling dimension, (should divide imageDim-filterDim+1)
%conv.strides    -    Strides for convolution
%conv.padding    -    Calibrated to preserve size

%full:
%full.outputSize - number of activations after passing through the layer


% Map the params (a vector into a stack of weights)

depth = numel(full);
stack = cell(depth,1);

% mark current position in parameter vector
cur_pos = 1;

for d = 1:numel(conv)
    
   stack{d} = struct;
   
   wlen = conv{d}.filterDim^2 * conv{d}.filterDepth * conv{d}.numFilters;
   blen = conv{d}.numFilters;
   
   if  conv{d}.filterDepth >1
       
        stack{d}.W = reshape(params(cur_pos : cur_pos+wlen-1), conv{d}.filterDim,...
        conv{d}.filterDim, conv{d}.filterDepth, conv{d}.numFilters);
        cur_pos = cur_pos + wlen ;
        stack{d}.b = params(cur_pos : cur_pos + blen -1);
        cur_pos = cur_pos + blen ;
  
   else
       
       stack{d}.W = reshape(params(cur_pos : cur_pos+wlen-1), conv{d}.filterDim,...
        conv{d}.filterDim, 1, conv{d}.numFilters);
        cur_pos = cur_pos + wlen;
        stack{d}.b = params(cur_pos : cur_pos + blen -1);
        cur_pos = cur_pos+ blen ;
   end
   
        output_dim = input_size(1) - conv{d}.filterDim + 2*(conv{d}.padding)/ conv{d}.strides +1;
        output_dim = output_dim/ conv{d}.poolDim;
        output_size = [output_dim, output_dim,conv{d}.numFilters];
        input_size = output_size;
end

 
for d = (numel(conv) + 1) : (numel(full))
    
    if length(input_size) > 1
        input_length = input_size(1) * input_size(2) * input_size(3);
    else
        input_length = input_size;
    end
    
    wlen = full{d}.outputSize * input_length;
    blen = full{d}.outputSize;
    
    stack{d}.W = reshape(params(cur_pos:cur_pos+ wlen-1),...
    full{d}.outputSize,input_length) ;
    cur_pos = cur_pos + wlen ;
    stack{d}.b = params(cur_pos : cur_pos + blen-1);
    cur_pos = cur_pos + blen ;
    input_size = full{d}.outputSize;  
end

end