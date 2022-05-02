%Main
%This code runs the TBP algorithm and each time picking different trainning and test sets

%%Read the data
clear; clc
load('mnist.mat');
trainX = double(trainX');
testX = double(testX');
[num_features,num_samples] = size(trainX);
trainY = trainY +1;
testY = testY +1;
trainYPred = trainY;
testYPred = testY;
trainY = numbers_to_labels(trainY);
testY = numbers_to_labels(testY);
%% devide the data to lables
devided_data_train={};
devided_data_test = {};
for i=1:50000
    devided_data_train_temp(i)=find(trainY(:,i)==1);
end
for i = 1:10
    devided_data_train{i}= find(devided_data_train_temp==i);
end
for i=1:10000
    devided_data_test_temp(i)=find(testY(:,i)==1);
end
for i = 1:10
    devided_data_test{i}= find(devided_data_test_temp==i);
end
%% normelize
trainX = (trainX-mean(trainX))./std(trainX);
testX = (testX-mean(testX))./std(testX);
%% Architecture
SizeInput = num_features;
meanRun =1;
number_of_iteration = 150;
num_from_each_lable = 500;
Sequence_Order = [0:9];
Number_of_roots = 16 ;
Test_each = 5;
numTest = 500;

%% Parameters
Mue =0.945;
eta = 0.0073;
Reg =  0.99999998;
Amp=0.001;

%% initialize the variables

devideTheNeuron = reshape([1:1:SizeInput],Number_of_roots,SizeInput/Number_of_roots);

for MEAN = 1:meanRun
    
    classification_Temp = [];
    devideTheNeuron = reshape(1:SizeInput,Number_of_roots,SizeInput/Number_of_roots);
    devideTheNeuronBinart = zeros(SizeInput,SizeInput/Number_of_roots);
    
    for i = 1 : SizeInput/Number_of_roots
        devideTheNeuronBinart(devideTheNeuron(:,i),i)=1;
    end
    
    [W1,W2] = weightsGenerator( SizeInput,SizeInput/Number_of_roots);
    b_1 = -1.*ones(SizeInput/Number_of_roots,10);
    b_2 = -1.*ones(1,10);
    
    Output_hidden_active_Temp = zeros(SizeInput/Number_of_roots,10) ;
    [Balanced_data_train] = Balanced_data_generator(devided_data_train,Sequence_Order,num_from_each_lable);
    
    for Iteration = 1:number_of_iteration % repeat learning of all the data.
        
        OrgenaizedIndexes = reshape(Balanced_data_train',1,numel(Balanced_data_train)); % different order of example for each iteration
        trainBatch = trainX(:,OrgenaizedIndexes);
        trainLabels = trainY(:,OrgenaizedIndexes);
        
        for TrainExample = 1 : num_from_each_lable*10 % going through all the data in a balanced way . 
            
            
            trainBatchNewTemp = squeeze(trainBatch(:,TrainExample,:)).*2 -1 ;
            Input_train =  trainBatchNewTemp.*devideTheNeuronBinart;
            
            Z1SaveAll =[];
            
            for System = 1 : 10 %calculating Z1 = X*W1 for each of the 10 systems . 
                
                Z1SaveAll(:,System) = (W1(:,System)'*Input_train)';
                
            end
            
            
            Output_hidden_layerNewTemp = Z1SaveAll+ b_1;
            Output_hidden_layerNew=Output_hidden_layerNewTemp-Amp*Output_hidden_active_Temp/((num_from_each_lable*10).*(Iteration-1) + TrainExample);
            Output_hidden_active_Temp=Output_hidden_active_Temp+Output_hidden_layerNewTemp;
            A1 = 1./(1+exp(-Output_hidden_layerNew));
            
            
            Z2 =  sum(W2.*A1)+ b_2;
            A2 = 1./(1+exp(-Z2));
            
            %Backprop:
            for System = 1:10
                
                outputDelta(:,System) = (A2(System)' - trainLabels(System,TrainExample));
                hiddenDelta(:,System)= (W2(:,System)*outputDelta(System)).*A1(:,System).*(1-A1(:,System));
                
                grad_2(:,System) = A1(:,System)*outputDelta(System)';
                grad_1(:,System) = (hiddenDelta(:,System)'*Input_train')';
                grad_2_b(:,System) = sum(outputDelta(System),2);
                grad_1_b(:,System) = sum(hiddenDelta(:,System),2);
                
            end
            
            
            if TrainExample*Iteration==1
                
                V_2 =-eta.*grad_2;
                V_1  =-eta.*grad_1;
                bV_2 =-(eta.*grad_2_b);
                bV_1 = -(eta.*grad_1_b);
                
            end
            
            
            V_2 = Mue*V_2 - (eta.*grad_2);
            V_1 = Mue*V_1 - (eta.*grad_1);
            
            bV_2 =Mue*bV_2-(eta.*grad_2_b);
            bV_1 = Mue*bV_1-(eta.*grad_1_b);
            
            W2 = Reg*W2 + V_2;
            W1 = Reg*W1 + V_1;
            
            b_2 = b_2 + bV_2;
            b_1 = b_1 + bV_1;
            
        end
        
        if mod(Iteration,Test_each)==0
            [Balanced_data_test] = Balanced_data_generator(devided_data_test,Sequence_Order,numTest);
            Output_hidden_active_TempTest = zeros(SizeInput/Number_of_roots,10) ;
            
            for j =  1 : numTest
                
                for t = 1 : 10
                    
                    TestBE = testX(:,Balanced_data_test(j,t)).*2 -1;
                    TestL = testY(:,Balanced_data_test(j,t));
                    TestBTemp = repmat(TestBE,1,SizeInput/Number_of_roots);
                    TestB = TestBTemp.*devideTheNeuronBinart;
                    
                    
                    for System = 1 : 10
                        
                        Z1SaveAll(:,System) = (W1(:,System)'*TestB)';
                        
                    end
                    
                    
                    Output_hidden_layerNewTemp = Z1SaveAll+ b_1;
                    Output_hidden_layerNew = Output_hidden_layerNewTemp -Amp*Output_hidden_active_TempTest/((10).*(j-1) + t);
                    Output_hidden_active_TempTest=Output_hidden_active_TempTest+Output_hidden_layerNewTemp;
                    A1 = 1./(1+exp(-Output_hidden_layerNew));
                    
                    Z2 =  sum(W2.*A1)+ b_2;
                    A2 = 1./(1+exp(-Z2));
                    
                    Max_outputTotalTemp = A2==max(A2);
                    Max_outputTotal = Max_outputTotalTemp.*(sum(Max_outputTotalTemp)==1); % check that there is only one correct answer
                    
                    classification_Temp(t,j) = sum(sum(double(Max_outputTotal)'.*TestL));
                    
                    
                end
            end
            
            classification_Test = mean(sum(classification_Temp,2)./numTest);
            
        end
    end
end


