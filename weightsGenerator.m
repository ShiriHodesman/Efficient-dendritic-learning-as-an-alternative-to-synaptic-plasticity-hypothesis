function [ W_temp_1, W_temp_2] = weightsGenerator( SizeInput,SizeHiddenLayer )

% generate random weights and normilazed them 
 
W_temp_1 = zeros(SizeInput,10);
W_temp_2 = zeros(SizeHiddenLayer,10);
for p =1:10
    TempR = randn(SizeInput, 1);
    W_temp_1(:,p) = (TempR-mean(TempR))./std(TempR);
    
    TempR2 = rand(SizeHiddenLayer,1);
    W_temp_2(:,p) = (TempR2-mean(TempR2))./std(TempR2);
end

  
end

