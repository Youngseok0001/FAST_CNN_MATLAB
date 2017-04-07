function [opttheta, cost] = minFuncAdaGrad(funObj,theta,data,labels,...
                        options)

% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  data       -  stores data in m x n x numExamples tensor
%  labels     -  corresponding labels in numExamples x 1 vector
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%  cost       -  cost of every iteration

%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0.9

%%======================================================================
%% Setup

epochs = options.epochs;
alpha = options.alpha;
fudge_factor = 1e-8;
minibatch = options.minibatch;
m = length(labels); % training set size
H_grad =  zeros(size(theta));
%%======================================================================
%% SGD loop
it = 0;
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1 : minibatch : m-minibatch+1
        it = it + 1;

        % get next randomly selected minibatch
        
        if numel(size(data)) == 3
        mb_data = gpuArray(data(:,:,rp(s:s+minibatch-1)));
        
        else
        mb_data = gpuArray(data(:,:,:,rp(s:s+minibatch-1)));
        end

        mb_labels = labels(rp(s:s+minibatch-1));

        % evaluate the objective function on the next minibatch
        [cost(it), grad] = funObj(theta,mb_data,mb_labels);
        
        cost(it) = cost(it)./minibatch;
        
        grad = grad./minibatch;
        
        H_grad = H_grad + grad.^2;
 
        theta = theta - (alpha ./( fudge_factor + sqrt(H_grad))).*grad;
        
        
               
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost(it));
    end;
        
end;

opttheta =  theta;

end
