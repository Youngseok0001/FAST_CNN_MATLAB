function [convolved_output] = wino_conv(data, filters, conv) 

if size(data,1) == 2

m = 2; r = 3;

B_T = [1 0 -1 0;
    0 1 1 0; 
    0 -1 1 0;
    0 1 0 -1];
B = B_T';

G = [1 0 0;
    0.5 0.5 0.5;
    0.5 -0.5 0.5;
    0 0 1];

G_T = G';

A_T = [1 1 1 0;
      0 1 -1 -1];
  
A = A_T';
    
else
    
m = 4; r = 3;

B_T = [4 0 -5 0 1 0;
    0 -4 -4 1 1 0; 
    0 4 -4 -1 1 0;
    0 -2 -1 2 1 0;
    0 2 -1 -2 1 0;
    0 4 0 -5 0 1];

B = B_T';

G = [1/4   0   0 ;
    -1/6 -1/6 -1/6 ;
    -1/6  1/6 -1/6 ;
    1/24  1/12 1/6;
     1/24 -1/12 1/6;
     0      0    1] ;
 
G_T = G';

A_T = [1 1 1 1 1 0;
      0 1 -1 2 -2 0;
      0 1  1 4  4 0;
      0 1 -1 8 -8 1];

A = A_T';

end


alpha =  m + r -1; %input tile size
data_padded = padarray(data, [conv.padding conv.padding]);
%P = (( (size(data,1) - alpha) + (2 * 1) )/4 + 1) ^2 ; % number of image tiles per channel per #images       
d = im2col_4D_sliding_v1(data_padded, [alpha alpha] , [4 4]);
d = permute(d,[1 3 2 4]);
d = reshape (d, alpha, alpha,  size(data,3), []);
g = permute(filters,[1 2 4 3]);



u = mul(G, g, G_T);
U  = permute(u,[3 4 1 2]);


v = mul(B_T, d , B);
V = permute(v,[3 4 1 2]);


M = pagefun(@mtimes,U,V);
m = permute(M, [3 4 1 2]);

Y = mul(A_T , m , A);


Y = permute(reshape(Y,size(Y,1),size(Y,2),size(Y,3),[],size(data,4)),[1 2 4 3 5]); 
convolved_output = col2im_4D_sliding_v1(Y);


end

