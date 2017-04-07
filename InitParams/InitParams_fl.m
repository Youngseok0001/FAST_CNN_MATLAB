function [ w, b, output_size] = InitParams_fl(input_size, spec)

if length(input_size) >1
    
    input_length = input_size(1) * input_size(2) *input_size(3);
    
else    
   input_length = input_size;
end

output_size = spec.outputSize;

w =sqrt(2/(input_length)) * randn (spec.outputSize, input_length);
b = zeros(spec.outputSize,1);

end

