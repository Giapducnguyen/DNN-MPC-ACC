clear all
close all
clc

load('trainingData_August_28_Trial_01.mat','x_nn_train_v','y_nn_train_v')

maxTrials_primal_fixedStructure_v = 3;  % training times for 1 fixed NN structure
maxTrials_primal_changedStructure_v = 5;    % maximum structure changing
net_primal_v = cell(1,1);

trialNumber_addNeur_v = 1;  % counter

% % Store all runs
netPrimal_all_v = cell(maxTrials_primal_fixedStructure_v,maxTrials_primal_changedStructure_v);
netPrimal_perf_all_v = nan(maxTrials_primal_fixedStructure_v,maxTrials_primal_changedStructure_v);

% % Initial primal network parameters
tr_mse_v = inf;
primalNetworkSize_v = [5 5 3];
tmp_v = sqrt(sum(y_nn_train_v.^2,1));
eps_mse_v = 1e-5*max(tmp_v);

%% Start training
while (tr_mse_v > eps_mse_v)&&(trialNumber_addNeur_v <= maxTrials_primal_changedStructure_v)
    
    disp(['network size: ', num2str(primalNetworkSize_v)]);
    trialNumber_fixedStruct_v = 0;
    for ii = 1 : maxTrials_primal_fixedStructure_v
        trialNumber_fixedStruct_v = trialNumber_fixedStruct_v + 1;
        disp(['trial number with fixed structure: ', num2str(trialNumber_fixedStruct_v)]);
        % % Define the neural network
        net_primal_v = feedforwardnet(primalNetworkSize_v);
        % net_primal_v = cascadeforwardnet(primalNetworkSize_v);
        net_primal_v.layers{1}.transferFcn = 'poslin';
        net_primal_v.layers{2}.transferFcn = 'poslin';
        net_primal_v.layers{3}.transferFcn = 'poslin';
%         net_primal_v.layers{4}.transferFcn = 'poslin';
        % % Dividing parameters for training, validating, testing
        net_primal_v.divideParam.trainRatio = 70/100;
        net_primal_v.divideParam.valRatio = 10/100;
        net_primal_v.divideParam.testRatio = 20/100;
        % % Number of epochs
        net_primal_v.trainParam.epochs = 500;
        % % view net
%         view(net_primal_v);
        % net_primal_v.trainFcn = 'trainscg';
        % % training
        [net_primal_v, tr_v] = train(net_primal_v,x_nn_train_v,y_nn_train_v,'useGPU','no');
        % % obtaining the MSE for the best epoch
        tr_mse_v = tr_v.perf(tr_v.best_epoch); % +1
        netPrimal_perf_all_v(trialNumber_fixedStruct_v,trialNumber_addNeur_v) = tr_mse_v;
        netPrimal_all_v{trialNumber_fixedStruct_v,trialNumber_addNeur_v} = net_primal_v;
        % % Display
        disp(['training error: ', num2str(tr_mse_v)]);
        disp(['best training error so far: ', num2str(min(min(netPrimal_perf_all_v)))]);
        % % check if the desired mse has been reached
        if tr_mse_v <= eps_mse_v
            disp(['----> Success!   | network size: ',num2str(primalNetworkSize_v), '    |  MSE: ', num2str(tr_mse_v)]);
            break
        end
    end
    primalNetworkSize_v = primalNetworkSize_v + 5;
    trialNumber_addNeur_v = trialNumber_addNeur_v + 1;
    disp(['Neuron Increase Index: ', num2str(trialNumber_addNeur_v)]);
end

%% Pick the best trained network (with lowest MSE)
min_mse_v = min(min(netPrimal_perf_all_v));
[idx1_v, idx2_v] = find(netPrimal_perf_all_v==min_mse_v);
net_primal_v = netPrimal_all_v{idx1_v,idx2_v};

% gen MATLAB function
genFunction(net_primal_v,'primalNet_vFcn_A28','MatrixOnly','yes');

% export to c code with mex interface
inputType = coder.typeof(double(0),[6 1]); % specify input type
codegen primalNet_vFcn_A28.m -config:mex -o pNet_v_CodeGen_A28 -args inputType

% save all data
save('primalNet_training_velocity_August_28_Trial_01.mat')

% call script to train space net
Step3_approxMPC_primalNet_training_space