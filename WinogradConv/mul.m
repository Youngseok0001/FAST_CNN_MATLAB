function [ OUTPUT ] = mul( Q ,B, Q_T )

B_mod = reshape(B,size(Q,2), []);
temp = reshape( Q * B_mod,size(Q,1),size(Q,2),[]);
temp  = reshape(permute(temp,[2 1 4 3]),size(Q_T,1),[]); 
OUTPUT = permute(reshape(Q * temp ,size(Q,1),size(Q_T,2),size(B,3),[]),[2 1 3 4]);

end

