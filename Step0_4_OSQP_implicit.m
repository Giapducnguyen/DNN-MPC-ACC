% ACC problem - implicit MPC using OSQP
clear all; close all; clc

%% Tuning parameters

% % Objective weighting matrices
Qe_v = diag([1,0.5]);    % 3; 0.5 achieving set velocity 

Qe_s = diag([1,0.5]);    % 3; 0.5 maintaining safe distance

R = 1; %250;               

% slack variables' weights
mu_s = 1e2;
mu_v = 1e3;

% % Prediction horizon
Np = 20;

% % preceding vehicle initial states
xp0 = 50; vp0 = 25; ap0 = 0;

% % following(ACC) vehicle initial states
xf0 = 10; vf0 = 20; af0 = 0;

%% Time data
Tsim = 120;              % Total running time
Ts = 0.1;               % sampling time
Nsim = floor(Tsim/Ts);  % number of time steps
Time_log = 0:Ts:Tsim;   % run time vectors

%% Constraint data
v_set = 30;             % driver's desired velocity
d0 = 10;                % stopping distance
tau_h = 1.4;            % constant time headway

max_d = Inf; %115;            % maximum sensing range
min_d_v = d0;           % safe distance
min_d_s = 0;            % no collision
max_v = 50;             % maximum velocity ~ 130 km/h imposed by law
min_v = 0;              % minimum velocity ~ no negative vel. on highways
g_acc = 9.81;           % gravitional acceleration
max_u = 2;              % maximum acceleration
min_u = -3;             % minimum acceleration

max_du = 0.2;           % maximum acceleration rate
min_du = -0.2;          % minimum acceleration rate

%% Model data
% Bandwidth of the lower-level controller
tau_f = 0.5;            % ACC vehicle    
tau_p = 0.5;            % Preceding vehicle      

% % Model matrices

% % State: [distance between 2 vehicles; ACC's velocity; ACC's accel.]
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

Bw = [Ts;  0;   0];             % disturbance matrix

Cd_v = [0, 1, 0;
        0, 0, 1];               % for velocity tracking
Cd_s = [1, 0, 0;
        0, 0, 1];               % for safe distance tracking

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
            zeros(nr,size(Az,2)),     zeros(nr,nu),   eye(nr)                   ]; % for velocity control
Az_ext_s = [Az,                             Bz,             zeros(size(Az,1),nr);
            zeros(nu,size(Az,2)),           eye(nu),        zeros(nu,nr);
            [0,0,tau_h*Ts,0;0,0,0,0],       zeros(nr,nu),   eye(nr)             ]; % for space control
Bz_ext = [Bz; eye(nu); zeros(nr,nu)];

E_v = [Cz_v, zeros(size(Cz_v,1),nu), -eye(nr)];
E_s = [Cz_s, zeros(size(Cz_s,1),nu), -eye(nr)];

%% Constraints on control U
Gu = [1;-1]; 
gu = [max_u;-min_u];

Gz_ext = Gu*[zeros(nu,nx), zeros(nu,nd), eye(nu), zeros(nu,nr)];
gz_ext = gu;

%% Constraints on states x --> z --> z_ezt
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

%% Constraints on dU (incremental control/ control rate)
Ku = [1;-1];
ku = [max_du; -min_du];

%% MPC formulation

n_epsilon = 1;

% % Objective weighting matrices
Q_v = E_v'*Qe_v*E_v;
QN_v = 1*Q_v;

Q_s = E_s'*Qe_s*E_s;
QN_s = 1*Q_s;

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

%% First-formualtion of OSQP object with initial x0
% Initial states
param0_v = [xp0-xf0;vf0;af0;vp0;0;[v_set;0]];
param0_s = [xp0-xf0;vf0;af0;vp0;0;[d0+tau_h*vf0;0]];

% Linear term in the cost functions
q_v = B_bar_v'*Q_bar_v*A_bar_v*param0_v;
q_s = B_bar_s'*Q_bar_s*A_bar_s*param0_s;

% Controller objects
controller = osqp;
% Object settings
settings = controller.default_settings();
settings.warm_start = true;
settings.verbose = true;

% Controller objects
controller_v = osqp;
controller_s = osqp;
% Object settings
settings_v = controller_v.default_settings();
settings_s = controller_s.default_settings();
settings_v.warm_start = true;
settings_v.verbose = true;
settings_s.warm_start = true;
settings_s.verbose = true;

controller_v.setup(H_v,q_v,Uad_v.A,[],Uad_v.b-Uad_v.B*param0_v,settings_v);
controller_s.setup(H_s,q_s,Uad_s.A,[],Uad_s.b-Uad_s.B*param0_s,settings_s);

% Export to C code
% controller_v.codegen('codegen_v','mexname','emosqp_v','force_rewrite',true);
% controller_s.codegen('codegen_s','mexname','emosqp_s','force_rewrite',true);

%% mpcActiveSetSolver options

options = mpcActiveSetOptions;
iA0_v = false(size(Uad_v.b));
iA0_s = false(size(Uad_s.b));

%% Main Simulations

% % Initialization loggings

% % preceding vehicle
xp_log = zeros(nx,Nsim+1);
xp_log(:,1) = [xp0; vp0; ap0];

% % following(ACC) vehicle
% Implicit
xf_log_impl = zeros(nx,Nsim+1);
xf_log_impl(:,1) = [xf0; vf0; af0];

% % Intervehicle states
% Implicit
xInter_log_impl = zeros(nx,Nsim+1);
xInter_log_impl(:,1) = [xp_log(1,1)-xf_log_impl(1,1); xf_log_impl(2,1); xf_log_impl(3,1)];

% % Execution time
% Implicit
executionTime_impl = zeros(1,Nsim);

% % ACC vehicle control logging
% Implicit
uf_log_impl = zeros(nu,Nsim);
duf_log_impl = zeros(nu,Nsim);

% % safe distance logging
% Implicit
safeDistance_impl = zeros(1,Nsim+1);
safeDistance_impl(1) = d0 + tau_h*xf_log_impl(2,1);

% % Disturbance estimate logging
% < will be developed later >

% % velocity control activation time record
% Implicit
velocityControlTime_impl = zeros(1,Nsim);

% Set velocity reference
v_ref = zeros(1,Nsim);

% infeasible problem counter
infeas_v = zeros(Nsim,1);
infeas_s = zeros(Nsim,1);

% salck variable records
slack_logging_v = zeros(n_epsilon,Nsim);
slack_logging_s = zeros(n_epsilon,Nsim);

% blend mode record
blend = zeros(Nsim,1);

% % main loop
for k = 1:Nsim
    % % Reference updates

    % safe distance - Implicit
    d_ref_impl = d0 + tau_h*xf_log_impl(2,k);
    safeDistance_impl(k+1) = d0 + tau_h*xf_log_impl(2,k);

    % velocity ref.
%     v_ref = v_set;
    if k <= 300
        v_ref(k) = 25; %v_set;
    elseif k <= 550 %550
        v_ref(k) = 15; %25;
    else
        v_ref(k) = 32;
    end

    % % preceding vehicle acceleration command
%     up_curr = 0.3*(1+k/300)*sin(0.2*(k-1)*Ts);

    if k == 1
        up_curr = 0;
    elseif k <= 100
        up_curr = -1; %-0.1*(k*Ts);
    elseif k <= 200
        up_curr = -1.5;
    elseif k <= 400
        up_curr = 0; %(-0.225)*(k*Ts) + 6;
    elseif k <= 600
        up_curr = 2; %-3;
    elseif k <= 750
        up_curr = 0; %3/8.8*(k*Ts) - 225/11;
    elseif k <= 925 %925
        up_curr = -1.5; %-1.5; %0.5; %0.1*(k*Ts)-6; %0.2*(k*Ts)-14; %0.1*(k*Ts)-6;
    else
        up_curr = 0;
    end

    % % previous control
    if k == 1
        u_prev_impl = 0;
    else
        u_prev_impl = uf_log_impl(:,k-1);
    end
    
    % Update OSQP Objects
    % New initial states
    param0_v_new = [xInter_log_impl(:,k);xp_log(2,k);u_prev_impl;[v_ref(k);0]];
    param0_s_new = [xInter_log_impl(:,k);xp_log(2,k);u_prev_impl;[d0+tau_h*xf_log_impl(2,k);0]];
    % Linear cost terms
    q_v_new = B_bar_v'*Q_bar_v*A_bar_v*param0_v_new;
    emosqp_v('update_lin_cost',q_v_new);
    q_s_new = B_bar_s'*Q_bar_s*A_bar_s*param0_s_new;
    emosqp_s('update_lin_cost',q_s_new);
    % Upper bounds
    ub_v_new = Uad_v.b-Uad_v.B*param0_v_new;
    emosqp_v('update_upper_bound',ub_v_new);
    ub_s_new = Uad_s.b-Uad_s.B*param0_s_new;
    emosqp_s('update_upper_bound',ub_s_new);

    % % control calculation
    % % Implicit
    %
%     if k == 1
%         slack_logging_s_prev = 0;
%     else
%         slack_logging_s_prev = slack_logging_s(:,k-1);
%     end
    %{
    [res_prim_s,res_dual_s,status_val_s,iter_s,run_time_s] = emosqp_s('solve');
    [res_prim_v,res_dual_v,status_val_v,iter_v,run_time_v] = emosqp_v('solve');
    if status_val_s == 1
        slack_logging_s(:,k) = res_prim_s(end-n_epsilon+1:end,1);
    end
    if status_val_v == 1
        slack_logging_v(:,k) = res_prim_v(end-n_epsilon+1:end,1);
    end

    if xInter_log_impl(1,k) <= safeDistance_impl(k+1) %+ slack_logging_s(:,k)
%         slack_logging_s(:,k) = res_prim_s(end-n_epsilon+1:end,1);
        executionTime_impl(k) = run_time_s;
        velocityControlTime_impl(1,k) = 0;
        if status_val_s ~= 1
            du_impl = 0; %-0.005*(-xInter_log_impl(1,k)+safe_dist); %0;
            infeas_s(k) = 1; 
        else
            du_impl = res_prim_s(1,1);
        end
    else
%         slack_logging_v(:,k) = res_prim_v(end-n_epsilon+1:end,1);
        executionTime_impl(k) = run_time_v;
        velocityControlTime_impl(1,k) = 1;
        if status_val_v ~= 1
            du_impl = 0; %10*(v_ref(k)-xf_log_impl(2,k)); %0;
            infeas_v(k) = 1; 
        else
            du_impl = res_prim_v(1,1);
        end
    end
    %}
    
    %
    if xInter_log_impl(1,k) <= safeDistance_impl(k+1) %+ slack_logging_s_prev% space control
        velocityControlTime_impl(1,k) = 0;
%         param0_s = [xInter_log_impl(:,k);xp_log(2,k);u_prev_impl;[d0+tau_h*xf_log_impl(2,k);0]];
%         q_s = B_bar_s'*Q_bar_s*A_bar_s*param0_s;
%         tic
%         [dU,exitflag,iA0_s,~] = mpcActiveSetSolver(H_s,q_s,Uad_s.A,Uad_s.b-Uad_s.B*param0_s,Uad_s.Ae,Uad_s.be,iA0_s,options);
%         [dU_s,exitflag_s,iA0_s] = spaceMPC(param0_s, iA0_s, H_s, q_s, Uad_s);
%         [dU_s,exitflag_s,iA0_s] = spaceMPC_mex(param0_s, iA0_s, H_s, q_s, Uad_s);
        [res_prim_s,res_dual_s,status_val_s,iter_s,run_time_s] = emosqp_s('solve');
        executionTime_impl(k) = run_time_s; %toc;

        if status_val_s ~= 1 % exitflag_s <= 0
            du_impl = 0;
            infeas_s(k) = 1; 
        else
            du_impl = res_prim_s(1,1); %dU_s(1,1);
            slack_logging_s(:,k) = res_prim_s(end-n_epsilon+1:end,1) ;%dU_s(end-n_epsilon+1:end,1);
        end
    else % velocity control
        velocityControlTime_impl(1,k) = 1;
%         param0_v = [xInter_log_impl(:,k);xp_log(2,k);u_prev_impl;[v_ref(k);0]];
%         q_v = B_bar_v'*Q_bar_v*A_bar_v*param0_v;
%         tic
%         [dU,exitflag,iA0_v,~] = mpcActiveSetSolver(H_v,q_v,Uad_v.A,Uad_v.b-Uad_v.B*param0_v,Uad_v.Ae,Uad_v.be,iA0_v,options);
%         [dU_v,exitflag_v,iA0_v] = velocityMPC(param0_v, iA0_v, H_v, q_v, Uad_v);
%         [dU_v,exitflag_v,iA0_v] = velocityMPC_mex(param0_v, iA0_v, H_v, q_v, Uad_v);
        [res_prim_v,res_dual_v,status_val_v,iter_v,run_time_v] = emosqp_v('solve');
        executionTime_impl(k) = run_time_v; %toc;
        if status_val_v ~= 1 %exitflag_v <= 0
            du_impl = 0;
            infeas_v(k) = 1; 
        else
            du_impl = res_prim_v(1,1); %dU_v(1,1);
            slack_logging_v(:,k) = res_prim_v(end-n_epsilon+1:end,1); %dU_v(end-n_epsilon+1:end,1);
        end
    end
    %}

    %{
    % blend mode
    timeStep = 10; % 10 steps = 1 second
    dist_thres = 5; % 10 meter
    if k > 10 && (xInter_log_impl(1,k-timeStep) - xInter_log_impl(1,k))/(timeStep*Ts) >= dist_thres % 15 (m/s)
        blend(k) = 1;
        [res_prim_s,res_dual_s,status_val_s,iter_s,run_time_s] = emosqp_s('solve');
        [res_prim_v,res_dual_v,status_val_v,iter_v,run_time_v] = emosqp_v('solve');
        hv = 10;
        heavisideStep = 1/(1+ exp(-2*hv*(xInter_log_impl(1,k) - safeDistance_impl(k+1))));
        du_impl = (1-heavisideStep)*res_prim_s(1,1) + (heavisideStep)*res_prim_v(1,1);
%         du_impl = res_prim_s(1,1);
    end
    %}
    duf_log_impl(:,k) = max(min_du, min(du_impl, max_du));
    uf_log_impl(:,k) = max(min_u, min(u_prev_impl + du_impl, max_u));
    

    % % % State updates
    % % preceding vehicle
    xp_log(:,k+1) = precedingVehDyn(Ts,tau_p,xp_log(:,k),up_curr);
    % % following (ACC) vehicle
    % Implicit
    xf_log_impl(:,k+1) = indiVehDyn(Ts,tau_f,xf_log_impl(:,k),uf_log_impl(:,k));
    % % Intervehicle states
    % Implicit
    xInter_log_impl(:,k+1) = [xp_log(1,k+1)-xf_log_impl(1,k+1); xf_log_impl(2,k+1); xf_log_impl(3,k+1)];
    
%     % Update OSQP Objects
%     % New initial states
%     param0_v_new = [xInter_log_impl(:,k+1);xp_log(2,k+1);uf_log_impl(:,k);[v_ref;0]];
%     param0_s_new = [xInter_log_impl(:,k+1);xp_log(2,k+1);uf_log_impl(:,k);[d0+tau_h*xf_log_impl(2,k+1);0]];
%     % Linear cost terms
%     q_v_new = B_bar_v'*Q_bar_v*A_bar_v*param0_v_new;
%     emosqp_v('update_lin_cost',q_v_new);
%     q_s_new = B_bar_s'*Q_bar_s*A_bar_s*param0_s_new;
%     emosqp_s('update_lin_cost',q_s_new);
%     % Upper bounds
%     ub_v_new = Uad_v.b-Uad_v.B*param0_v_new;
%     emosqp_v('update_upper_bound',ub_v_new);
%     ub_s_new = Uad_s.b-Uad_s.B*param0_s_new;
%     emosqp_s('update_upper_bound',ub_s_new);

end

%% Plots
figure(1)
fig1 = tiledlayout(3,1);

nexttile
hold on
% plot(Time_log,xInter_log_NN(1,:),'LineWidth',1.5,'LineStyle','-','Color','m');
% plot(Time_log,safeDistance_NN,'LineWidth',1.5,'LineStyle','-.','Color','m');
plot(Time_log,xInter_log_impl(1,:),'LineWidth',1,'LineStyle','-','Color','m');
plot(Time_log,safeDistance_impl,'LineWidth',1,'LineStyle','--','Color','b');
ylabel('$d (m)$','Interpreter','latex');
title('Intervehicle Distance','Interpreter','latex');
% legend({'actual-DeepNN','safe-DeepNN','actual-implicit','safe-implicit'},'Interpreter','latex','Location','best');
legend({'actual-implicit','safe-implicit'},'Interpreter','latex','Location','best');
grid on; box on;
ax = gca; ax.FontSize = 14;

nexttile
hold on
plot(Time_log,xp_log(2,:),'LineWidth',1,'LineStyle','-','Color','g');
% plot(Time_log,xf_log_NN(2,:),'LineWidth',1.5,'LineStyle','-','Color','m');
plot(Time_log,xf_log_impl(2,:),'LineWidth',1,'LineStyle','--','Color','b');
% plot(Time_log,v_set*ones(size(Time_log)),'LineWidth',1.5,'LineStyle','--','Color','k');
plot(Time_log(1:end-1),v_ref,'LineWidth',1,'LineStyle','--','Color','k');
ylabel('$v (m/s)$','Interpreter','latex');
title('Velocity','Interpreter','latex');
% legend({'preceding','ACC-DeepNN','ACC-implicit','driver set'},'Interpreter','latex','Location','best');
legend({'preceding','ACC-implicit','driver set'},'Interpreter','latex','Location','best');
grid on; box on;
ax = gca; ax.FontSize = 14;

nexttile
hold on
plot(Time_log,xp_log(3,:),'LineWidth',1,'LineStyle','-','Color','g');
% plot(Time_log,xf_log_NN(3,:),'LineWidth',1.5,'LineStyle','-','Color','m');
plot(Time_log,xf_log_impl(3,:),'LineWidth',1,'LineStyle','--','Color','b');
plot(Time_log,min_u*ones(size(Time_log)),'LineWidth',1,'LineStyle','--','Color','k');
plot(Time_log,max_u*ones(size(Time_log)),'LineWidth',1,'LineStyle','--','Color','k');
ylabel('$a (m/s^2)$','Interpreter','latex');
xlabel('Time (seconds)','Interpreter','latex');
title('Acceleration','Interpreter','latex');
% legend({'preceding','ACC-DeepNN','ACC-implicit','constraints'},'Interpreter','latex','Location','best');
legend({'preceding','ACC-implicit','constraints'},'Interpreter','latex','Location','best');
grid on; box on;
ylim([-3.5 2.5]);
yticks([-3 0 2])
ax = gca; ax.FontSize = 14;

fig1.TileSpacing = 'compact';
fig1.Padding = 'compact';

set(gcf,"Units","points",'Position',[50, 50, 500, 650])

figure(2)
fig2 = tiledlayout(2,1);

nexttile
hold on
% stairs(Time_log(1:end-1),max(min_u,min(uf_log_NN,max_u)),'LineWidth',1.5,'LineStyle','-','Color','m');
stairs(Time_log(1:end-1),uf_log_impl,'LineWidth',1,'LineStyle','--','Color','b');
plot(Time_log(1:end-1),min_u*ones(size(Time_log(1:end-1))),'LineWidth',1,'LineStyle','--','Color','k');
plot(Time_log(1:end-1),max_u*ones(size(Time_log(1:end-1))),'LineWidth',1,'LineStyle','--','Color','k');
ylabel('$u_f (m/s^2)$','Interpreter','latex');
% xlabel('Time (seconds)','Interpreter','latex');
title('Acceleration Command','Interpreter','latex');
% legend({'$u_f$-DeepNN','$u_f$-implicit','constraints'},'Interpreter','latex','Location','best');
legend({'$u_f$-implicit','constraints'},'Interpreter','latex','Location','best');
grid on; box on;
ylim([-3.5 4]);
yticks([-3 0 2])
ax = gca; ax.FontSize = 14;

nexttile
hold on
% stairs(Time_log(1:end-1),max(min_du,min(duf_log_NN,max_du)),'LineWidth',1.5,'LineStyle','-','Color','m');
stairs(Time_log(1:end-1),duf_log_impl,'LineWidth',1,'LineStyle','--','Color','b');
plot(Time_log(1:end-1),min_du*ones(size(Time_log(1:end-1))),'LineWidth',1,'LineStyle','--','Color','k');
plot(Time_log(1:end-1),max_du*ones(size(Time_log(1:end-1))),'LineWidth',1,'LineStyle','--','Color','k');
ylabel('$\dot{u}_f (m/s^2)$','Interpreter','latex');
xlabel('Time (seconds)','Interpreter','latex');
title('Acceleration Rate Command','Interpreter','latex');
% legend({'$\dot{u}_f$-DeepNN','$\dot{u}_f$-implicit','constraints'},'Interpreter','latex','Location','best');
legend({'$\dot{u}_f$-implicit','constraints'},'Interpreter','latex','Location','best');
grid on; box on;
ylim([-0.22 0.4])
yticks([-0.2 0 0.2])
ax = gca; ax.FontSize = 14;

fig2.TileSpacing = 'compact';
fig2.Padding = 'compact';

set(gcf,"Units","points",'Position',[550, 50, 500, 325])


figure(3)
fig3 = tiledlayout(4,1);

% load('exeTime_mosek_H.mat')
% load('exeTime_osqp_H.mat')
% load('exeTime_qpoases_H.mat')

nexttile
hold on
% stairs(Time_log(1:end-1),executionTime_NN,'LineWidth',1.5,'LineStyle','-','Color','m');
stairs(Time_log(1:end-1),executionTime_impl,'LineWidth',1,'LineStyle','--','Color','b');
% stairs(Time_log(1:end-1),executionTime_impl_mosek,'LineWidth',1,'LineStyle','--','Color','k');
% stairs(Time_log(1:end-1),executionTime_impl_osqp,'LineWidth',1,'LineStyle','--','Color','g');
% stairs(Time_log(1:end-1),executionTime_impl_qpoases,'LineWidth',1,'LineStyle','--','Color','r');
ylabel('$t_{exe} (sec)$','Interpreter','latex');
title('Computation time','Interpreter','latex');
% legend({'DeepNN','GUROBI','MOSEK','OSQP','QPOASES'},'Interpreter','latex','Location','best');
% legend({'DeepNN','implicit'},'Interpreter','latex','Location','best');
grid on; box on;
ax = gca; ax.FontSize = 14;

nexttile
hold on
% stairs(Time_log(1:end-1),executionTime_NN,'LineWidth',1.5,'LineStyle','-','Color','m');
% stairs(Time_log(1:end-1),executionTime_impl,'LineWidth',2,'LineStyle',':','Color','b');
% stairs(Time_log(1:end-1),executionTime_impl_mosek,'LineWidth',1,'LineStyle','--','Color','k');
% stairs(Time_log(1:end-1),executionTime_impl_osqp,'LineWidth',1,'LineStyle','--','Color','g');
% stairs(Time_log(1:end-1),executionTime_impl_qpoases,'LineWidth',1,'LineStyle','--','Color','r');
plot(Time_log(1:end-1), infeas_v,'b-','LineWidth',1)
plot(Time_log(1:end-1), infeas_s,'m--','LineWidth',1)
% ylabel('$t_{exe} (sec)$','Interpreter','latex');
title('Infeasible OCP Flag','Interpreter','latex');
legend({'velocity mode','space mode'},'Interpreter','latex','Location','best');
grid on; box on;
% ylim([0 0.018])
ax = gca; ax.FontSize = 14;

% Slack variables
nexttile
hold on
plot(Time_log(1:end-1), slack_logging_v,'b-','LineWidth',1)
plot(Time_log(1:end-1), slack_logging_s,'m--','LineWidth',1)
ylabel('$d\,(m)$','Interpreter','latex');
title('Slack Variables','Interpreter','latex');
legend({'velocity mode','space mode'},'Interpreter','latex','Location','best');
grid on; box on;
ax = gca; ax.FontSize = 14;

% % Velocity activation time instance
nexttile
hold on
% plot(Time_log(1:end-1),velocityControlTime_NN,'LineWidth',2,'LineStyle','-','Color','m')
plot(Time_log(1:end-1),velocityControlTime_impl,'LineWidth',1,'LineStyle','-','Color','b')
plot(Time_log(1:end-1), blend,'m--','LineWidth',1)
ylabel('State','Interpreter','latex');
xlabel('Time (seconds)','Interpreter','latex');
title('Velocity Control Mode Activation Flag','Interpreter','latex');
legend({'velocity mode','blend mode'},'Interpreter','latex','Location','best');
grid on; box on;
ax = gca; ax.FontSize = 14;



fig3.TileSpacing = 'compact';
fig3.Padding = 'compact';

set(gcf,"Units","points",'Position',[850, 50, 500, 650])
%}

figure
hold on
plot(Time_log,xp_log(1,:),'k--','LineWidth',1)
plot(Time_log,xf_log_impl(1,:),'b-.','LineWidth',1)
ylabel('$\mathrm{Position} (m)$','Interpreter','latex');
xlabel('Time (seconds)','Interpreter','latex');
% plot(Time_log,xf_log_NN(1,:),'m-')
% legend({'preceding','implicit','DNN'})
legend({'preceding','implicit'},'Interpreter','latex','Location','best')
grid on; box on;
ax = gca; ax.FontSize = 14;

%% Function helpers

% % Function helper 1: Individual vehicle dynamics
function x_next = indiVehDyn(Ts,tau,x_curr,u_curr)
Ad = [1, Ts, 0;
      0, 1,  Ts;
      0, 0,  1-Ts/tau];
Bd = [0; 0; Ts/tau];
x_next = Ad*x_curr + Bd*u_curr;
end

function x_next = precedingVehDyn(Ts,tau,x_curr,u_curr)
Ad = [1, Ts, 0;
      0, 1,  Ts;
      0, 0,  1-Ts/tau];
Bd = [0; 0; Ts/tau];
x_curr(2) = max(0, x_curr(2));
x_next = Ad*x_curr + Bd*u_curr;
x_next(2) = max(x_next(2),0);
% if x_next(2) == 0
%     x_next(3) = 0;
% end
end

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