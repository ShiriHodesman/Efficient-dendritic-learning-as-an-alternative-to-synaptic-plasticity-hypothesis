function [y_new] = numbers_to_labels(y) 
%number_to_labels Vectorize each label
 y_new = zeros(max(y),length(y));
 for i=1:length(y)
    y_new(y(i),i) = 1;
 end
end