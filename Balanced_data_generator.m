function [Balanced_data] = Balanced_data_generator(devided_data_train,SeqOrder,numPairs)
%balance the data so that every step will contain a deifferent lable by the
%order that was set 
for i = 1 : size(SeqOrder,2)
    TempRE = devided_data_train{SeqOrder(i)+1};
    Balanced_data(:,i) = TempRE(randperm(length(TempRE),numPairs));
    
end 

end

