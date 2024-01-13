clear all
close all
clc

load('trainingData_August_28_Trial_01.mat','x_nn_train_s','y_nn_train_s')

maxTrials_primal_fixedStructure_s = 3;
maxTrials_primal_changedStructure_s = 5;
net_primal_s = cell(1,1);

trialNumber_addNeur_s = 1;

% % Store all runs
netPrimal_all_s = cell(maxTrials_primal_fixedStructure_s,maxTrials_primal_changedStructure_s);
netPrimal_perf_all_s = nan(maxTrials_primal_fixedStructure_s,maxTrials_primal_changedStructure_s);

% % Initial primal network parameters
tr_mse_s = inf;
primalNetworkSize_s = [5 5 3];
tmp_s = sqrt(sum(y_nn_train_s.^2,1));
eps_mse_s = 1e-5*max(tmp_s);

%% Start training
while (tr_mse_s > eps_mse_s)&&(trialNumber_addNeur_s <= maxTrials_primal_changedStructure_s)
    
    disp(['network size: ', num2str(primalNetworkSize_s)]);
    trialNumber_fixedStruct_s = 0;
    for ii = 1 : maxTrials_primal_fixedStructure_s
        trialNumber_fixedStruct_s = trialNumber_fixedStruct_s + 1;
        disp(['trial number with fixed structure: ', num2str(trialNumber_fixedStruct_s)]);
        % % Define the neural network
        net_primal_s = feedforwardnet(primalNetworkSize_s);
        % net_primal_s = cascadeforwardnet(primalNetworkSize_s);
        net_primal_s.layers{1}.transferFcn = 'poslin';
        net_primal_s.layers{2}.transferFcn = 'poslin';
        net_primal_s.layers{3}.transferFcn = 'poslin';
%         net_primal_s.layers{4}.transferFcn = 'poslin';
        % % Dividing parameters for training, validating, testing
        net_primal_s.divideParam.trainRatio = 70/100;
        net_primal_s.divideParam.valRatio = 10/100;
        net_primal_s.divideParam.testRatio = 20/100;
        % % Number of epochs
        net_primal_s.trainParam.epochs = 500;
        % % view net
%         view(net_primal_s);
        % net_primal_s.trainFcn = 'trainscg';
        % % training
        [net_primal_s, tr_s] = train(net_primal_s,x_nn_train_s,y_nn_train_s,'useGPU','no');
        % % obtaining the MSE for the best epoch
        tr_mse_s = tr_s.perf(tr_s.best_epoch); % +1
        netPrimal_perf_all_s(trialNumber_fixedStruct_s,trialNumber_addNeur_s) = tr_mse_s;
        netPrimal_all_s{trialNumber_fixedStruct_s,trialNumber_addNeur_s} = net_primal_s;
        % % Display
        disp(['training error: ', num2str(tr_mse_s)]);
        disp(['best training error so far: ', num2str(min(min(netPrimal_perf_all_s)))]);
        % % check if the desired mse has been reached
        if tr_mse_s <= eps_mse_s
            disp(['----> Success!   | network size: ',num2str(primalNetworkSize_s), '    |  MSE: ', num2str(tr_mse_s)]);
            break
        end
    end
    primalNetworkSize_s = primalNetworkSize_s + 5;
    trialNumber_addNeur_s = trialNumber_addNeur_s + 1;
    disp(['Neuron Increase Index: ', num2str(trialNumber_addNeur_s)]);
end

%% Pick the best trained network (with lowest MSE)
min_mse_s = min(min(netPrimal_perf_all_s));
[idx1_s, idx2_s] = find(netPrimal_perf_all_s==min_mse_s);
net_primal_s = netPrimal_all_s{idx1_s,idx2_s};

% gen MATLAB function
genFunction(net_primal_s,'primalNet_sFcn_A28','MatrixOnly','yes');

% export to c code with mex interface
inputType = coder.typeof(double(0),[6 1]); % specify input type
codegen primalNet_sFcn_A28.m -config:mex -o pNet_s_CodeGen_A28 -args inputType

% save all data
save('primalNet_training_space_August_28_Trial_01.mat')

% call script to test velocity net
Step4_approxMPC_primalNet_testing_velocity