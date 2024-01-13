% % Yalmip - ACC - tracking - without DO

clear all
close all
clc

%% Loading data
load("primalNet_training_space_August_24_Trial_01.mat",'net_primal_s')

%% Constraints
v_set = 30;     % driver's desired velocity
d0 = 10;                % stopping distance
tau_h = 1.4;            % constant time headway

max_d = 500;            % maximum sensing range
min_d_v = d0;              % safe distance
min_d_s = 0;              % no collision
max_v = 50; %v_set+0.2; %36.11;          % maximum velocity ~ 130 km/h imposed by law
min_v = 0;              % minimum velocity ~ no negative velocity on highways
g_acc = 9.81;           % gravitional acceleration
max_u = 2; %0.25*g_acc;     % maximum acceleration
min_u = -3; %-0.5*g_acc;     % minimum acceleration

max_du = 0.2;
min_du = -0.2; 

%% Time data
Tsim = 80;
Ts = 0.1;
Nsim = floor(Tsim/Ts);
Time_log = 0:Ts:Tsim;

%% Model data
% % Bandwidth of the lower-level controller
tau_f = 0.5;        % of ACC vehicle    
tau_p = 0.5;        % of preceding vehicle            

% % Model matrices

% % State: [distance between 2 vehicles; ACC's velocity; ACC's acceleration]
% % x(k) = [d(k); v_f(k); a_f(k)];
% % d(k) = s_p(k) - s_f(k);

% % Control: ACC's acceleration command
% % u(k) = u_f(k);

% % Measured Disturbance: preceding vehicle's velocity
% % w(k) = v_p(k);

% % Dynamics:
% % x(k+1) = Ad*x(k) + Bd*u(k) + Bw* w(k);
% % y(k)   = Cd_v*x(k);          for velocity control
% % y(k)   = Cd_s*x(k);          for space control

Ad = [1,  -Ts,  0;
      0,   1,   Ts;
      0,   0,   1-Ts/tau_f];
Bd = [0;   0;   Ts/tau_f];

Bw = [Ts;  0;   0];     % disturbance matrix

Cd_v = [0, 1, 0;
        0, 0, 1];   % for velocity tracking
Cd_s = [1, 0, 0;
        0, 0, 1];   % for safe distance tracking

[nx,nu] = size(Bd);
nd = 1;
nr = 2;

%% Extend state to include disturbance estimate \hat{w} 
% % state z = [x(k); \hat{w}(k)];

Az = [Ad,                       Bw;
      zeros(nd,size(Ad,2)),     eye(nd)];
Bz = [Bd; zeros(nd,size(Bd,2))];

Cz_v = [Cd_v, zeros(size(Cd_v,1),nd)];
Cz_s = [Cd_s, zeros(size(Cd_s,1),nd)];


nz = size(Az,2);

%% Extend state to include control increments du and reference r
% % state z = [z(k); u(k-1); r(k)];

Az_ext_v = [Az,                       Bz,             zeros(size(Az,1),nr);
          zeros(nu,size(Az,2)),     eye(nu),        zeros(nu,nr);
          zeros(nr,size(Az,2)),     zeros(nr,nu),   eye(nr)             ]; % for velocity control
Az_ext_s = [Az,                       Bz,             zeros(size(Az,1),nr);
          zeros(nu,size(Az,2)),     eye(nu),        zeros(nu,nr);
          [0,0,tau_h*Ts,0;0,0,0,0],     zeros(nr,nu),   eye(nr)             ]; % for space control
Bz_ext = [Bz; eye(nu); zeros(nr,nu)];

E_v = [Cz_v, zeros(size(Cz_v,1),nu), -eye(nr)];
E_s = [Cz_s, zeros(size(Cz_s,1),nu), -eye(nr)];


% % Constraints on control U
Gu = [1;-1]; 
gu = [max_u;-min_u];

Gz_ext = Gu*[zeros(nu,nx), zeros(nu,nd), eye(nu), zeros(nu,nr)];
gz_ext = gu;

% % Constraints on states x --> z --> z_ezt
Fx_v = [-1, tau_h,  0;
        0,  1,      0;
        0,  -1,     0;
        0,  0,      1;
        0,  0,      -1];
Fx_s = [1, -tau_h, 0;
     -1, 0, 0;
      0, 1, 0;
      0,-1, 0;
      0, 0, 1;
      0, 0,-1];
fx_v = [-min_d_v;max_v;-min_v;max_u;-min_u];
fx_s = [min_d_v;-min_d_s;max_v;-min_v;max_u;-min_u];

Fz_v = Fx_v*[eye(nx), zeros(nx,nd)];
Fz_s = Fx_s*[eye(nx), zeros(nx,nd)];
fz_v = fx_v;
fz_s = fx_s;

Fz_ext_v = Fz_v*[eye(nz), zeros(nz,nu), zeros(nz,nr)];
Fz_ext_s = Fz_s*[eye(nz), zeros(nz,nu), zeros(nz,nr)];
fz_ext_v = fz_v;
fz_ext_s = fz_s;

% % Constraints on dU (increment control?)
Ku = [1;-1];
ku = [max_du; -min_du];

%% MPC data

n_epsilon = 1; % number of slack variable

% % Prediction horizon
Np = 20;

% % Objective weighting matrices

R = 1;

Qe_v = diag([1,0.5]);    % achieving set velocity, no penalty on acceleration 
Q_v = E_v'*Qe_v*E_v;
QN_v = Q_v;

Qe_s = diag([1,0.5]);    % maintaining safe distance, less acceleration
Q_s = E_s'*Qe_s*E_s;
QN_s = Q_s;

% slack variables' weights
mu_s = 1e2;
mu_v = 1e3;

%% QP formulation

Q_bar_v = blkdiag(kron(eye(Np-1),Q_v),QN_v);
R_bar_v = blkdiag(kron(eye(Np),R), mu_v*eye(n_epsilon));

Q_bar_s = blkdiag(kron(eye(Np-1),Q_s),QN_s);
R_bar_s = blkdiag(kron(eye(Np),R), mu_s*eye(n_epsilon));                

% % lifted matrices
[A_bar_v,B_bar_v] = genConMat(Az_ext_v,Bz_ext,Np,n_epsilon);
[A_bar_s,B_bar_s] = genConMat(Az_ext_s,Bz_ext,Np,n_epsilon);

% % Admissible inputs
Uad_v = AdmissibleInputs(Az_ext_v,Bz_ext,Np,Fz_ext_v,fz_ext_v,gz_ext,Gz_ext,Ku,ku,n_epsilon,nu);
Uad_s = AdmissibleInputs(Az_ext_s,Bz_ext,Np,Fz_ext_s,fz_ext_s,gz_ext,Gz_ext,Ku,ku,n_epsilon,nu);

% % QP matrices
H_v = B_bar_v'*Q_bar_v*B_bar_v + R_bar_v;
H_s = B_bar_s'*Q_bar_s*B_bar_s + R_bar_s;

%% mpcActiveSetSolver options

options = mpcActiveSetOptions;
iA0_v = false(size(Uad_v.b));
iA0_s = false(size(Uad_s.b));


%% Test the quality of the trained NNs
num_testRuns = 1e6;
% % Ideally have to pick with log over log formula
% % beta = 1e-7;
% % eps = 5e-2;
% % gam_tol = 0.1; % allowable suboptimality
% % num_testRuns = log(1/beta)/log(1/(1-eps));

% % Data logging Initialization
actual_gapTest_primal_s = nan(num_testRuns,1);
relative_gapTest_primal_s = nan(num_testRuns,1);

% optimalValueTest_v = nan(num_testRuns,1);
x_nn_test_s = nan(nx+nd+nu+1,num_testRuns);
y_nn_test_s = nan(Np*nu,num_testRuns);

U_test_s = nan(Np*nu+n_epsilon,num_testRuns);
num_infeas_s = 0;

%% Gridding

vf_set_vec = 5:1:45;
d_vec = min_d_s:0.5:max_d; % 0.5
vf_vec = min_v:0.1:max_v;
af_vec = min_u:0.02:max_u; % 0.02
vp_vec = min_v:0.1:max_v+20;
uf0_vec = min_u:0.02:max_u; % 0.02

%% Main test loop

jj = 1;
while jj <= num_testRuns
    % % Randomly select states & parameters
    i1 = randi([1, length(vf_set_vec)],1);   v_set = vf_set_vec(i1);
    i2 = randi([1, length(d_vec)],1);       dk = d_vec(i2);
    i3 = randi([1, length(vf_vec)],1);      vfk = vf_vec(i3);
    i4 = randi([1, length(af_vec)],1);      afk = af_vec(i4);
    i5 = randi([1, length(vp_vec)],1);      vpk = vp_vec(i5);
    i6 = randi([1, length(uf0_vec)],1);     ufk0 = uf0_vec(i6);
    % references
    ref_v = v_set;
    ref_s = d0+tau_h*vfk;
    ref_af = 0;
    
    if dk <= d0+tau_h*vfk
        % % store NN input
        param0_s = [dk;vfk;afk;vpk;ufk0;ref_s;ref_af];
        x_nn_test_s(:,jj) = param0_s(1:6);
    
        q_s = B_bar_s'*Q_bar_s*A_bar_s*param0_s;
        [dU_s,exitflag_s,iA0_s,~] = mpcActiveSetSolver(H_s,q_s,Uad_s.A,Uad_s.b-Uad_s.B*param0_s,Uad_s.Ae,Uad_s.be,iA0_s,options);

        if exitflag_s <= 0
            num_infeas_s = num_infeas_s + 1;
        else
            jj
            % % control policy by NN
            uNN_s = net_primal_s(param0_s(1:6));

            % % project to guarantee feasibility
            %{
            polfeas_s = Polyhedron('A',Uad_s.A,'b',Uad_s.b-Uad_s.B*param0_s);
            polfeas_s.minHRep();
            try
                U_test_s(:,jj) = polfeas_s.project(uNN_s).x;
            catch
                U_test_s(:,jj) = uNN_s;
            end
            % U_test_v(:,ii) = polfeas_v.project(uNN_v).x;
            %}
            U_test_s(:,jj) = uNN_s;
            % % objective value of online problem
            objective_onMPCTest_s = 0.5*dU_s'*H_s*dU_s + q_s'*dU_s;
            % % compute the cost using NN solution
            objective_approxMPCTest_s = 0.5*U_test_s(:,jj)'*H_s*U_test_s(:,jj) + q_s'*U_test_s(:,jj);
            % % Calculate the objective value gaps
            actual_gapTest_primal_s(jj,1) = norm(objective_onMPCTest_s-objective_approxMPCTest_s);
            relative_gapTest_primal_s(jj,1) = norm(objective_onMPCTest_s-objective_approxMPCTest_s)/norm(objective_onMPCTest_s);
            % % update counter
            jj = jj+1;
        end
    end
end

%% Suboptimality statistics
sA28_01 = sum(relative_gapTest_primal_s(:,1)<0.1/100)/num_testRuns*100;
sA28_05 = sum(relative_gapTest_primal_s(:,1)<0.5/100)/num_testRuns*100;
sA28_1 = sum(relative_gapTest_primal_s(:,1)<1/100)/num_testRuns*100;
sA28_5 = sum(relative_gapTest_primal_s(:,1)<5/100)/num_testRuns*100;
sA28_10 = sum(relative_gapTest_primal_s(:,1)<10/100)/num_testRuns*100;

disp(['online/offline gap rate below 0.1%: ' num2str(sum(relative_gapTest_primal_s(:,1)<0.1/100)/num_testRuns*100) '%'])
disp(['online/offline gap rate below 0.5%: ' num2str(sum(relative_gapTest_primal_s(:,1)<0.5/100)/num_testRuns*100) '%'])
disp(['online/offline gap rate below 1%: '   num2str(sum(relative_gapTest_primal_s(:,1)<1/100)/num_testRuns*100) '%'])
disp(['online/offline gap rate below 5%: '   num2str(sum(relative_gapTest_primal_s(:,1)<5/100)/num_testRuns*100) '%'])
disp(['online/offline gap rate below 10%: '   num2str(sum(relative_gapTest_primal_s(:,1)<10/100)/num_testRuns*100) '%'])

%% Function helper
% 
% % Function helper 2: augmented dynamics over the prediction horizon
function [A_bar,B_bar] = genConMat(A,B,Np,n_epsilon)

A_bar = cell2mat(cellfun(@(x)A^x,num2cell((1:Np)'),'UniformOutput',false));
B_bar = tril(cell2mat(cellfun(@(x)A^x,num2cell(toeplitz(0:Np-1)),'UniformOutput',false)))*kron(eye(Np),B);

% % or using cell
%
% A_bar = cell(Np, 1);
% B_bar = cell(Np,Np);
% b0 = zeros(size(B));
% for i = 1:Np
%     A_bar{i} = A^i;
%     for j = 1:Np
%         if i >= j
%             B_bar{i,j} = A^(i-j)*B;
%         else
%             B_bar{i,j} = b0;
%         end
%     end
% end
% A_bar = cell2mat(A_bar);
% B_bar = cell2mat(B_bar);
n_state = size(A,2);
B_bar = [B_bar, zeros(n_state*Np,n_epsilon)];
end

% Function helper 3: admissible inputs: satisfying state-input constraints
function Uad = AdmissibleInputs(A,B,Np,Fz_ext,fz_ext,gz_ext,Gz_ext,Ku,ku,n_epsilon,nu)

[A_bar,B_bar] = genConMat(A,B,Np,n_epsilon);

E = [zeros(n_epsilon,Np*nu), eye(n_epsilon)];
E_bar = [E; zeros(size(fz_ext,1)-n_epsilon, size(E,2))];
E_tilde = kron(ones(Np,1),E_bar);

Uad.A = [blkdiag(kron(eye(Np-1),Fz_ext),Fz_ext)*B_bar - E_tilde;kron(eye(Np),Gz_ext)*B_bar;[kron(eye(Np),Ku), zeros(Np*nu*2,n_epsilon)]];
Uad.b = [[kron(ones(Np-1,1),fz_ext);fz_ext];kron(ones(Np,1),gz_ext);kron(ones(Np,1),ku)];
Uad.B = [blkdiag(kron(eye(Np-1),Fz_ext),Fz_ext)*A_bar;kron(eye(Np),Gz_ext)*A_bar;zeros(size(kron(ones(Np,1),ku),1),size(A_bar,2))];
Uad.Ae = zeros(0,size(B_bar,2));
Uad.be = zeros(0,1);
end