% FORCE.m
%
% This function generates the sum of 4 sine waves in figure 2D using the architecture of figure 1A with the RLS
% learning rule.
%
% written by David Sussillo

disp('Clearing workspace.');
clear;

linewidth = 3;
fontsize = 14;
fontweight = 'bold';

N = 1000;
p = 0.1;
g = 1.5;				% g greater than 1 leads to chaotic networks.
alpha = 1.0;
nsecs = 1440;
dt = 0.1;
learn_every = 2;

tau = 30;

scale = 1.0/sqrt(p*N);
M = sprandn(N,N,p)*g*scale;
M = full(M);

nReadoutUnits = 3;
nRec2Out = N;
wo = zeros(nRec2Out,nReadoutUnits);
dw = zeros(nRec2Out,nReadoutUnits);
wf = 2.0*(rand(N,nReadoutUnits)-0.5);
%wf = zeros(N,nReadoutUnits);
%wf = repmat(2.0*(rand(N,1)-0.5),1,nReadoutUnits);

disp(['   N: ', num2str(N)]);
disp(['   g: ', num2str(g)]);
disp(['   p: ', num2str(p)]);
disp(['   nRec2Out: ', num2str(nRec2Out)]);
disp(['   alpha: ', num2str(alpha,3)]);
disp(['   nsecs: ', num2str(nsecs)]);
disp(['   learn_every: ', num2str(learn_every)]);


simtime = 0:dt:nsecs-dt;
simtime_len = length(simtime);
simtime2 = 1*nsecs:dt:2*nsecs-dt;

amp = 1.3;
freq = 1/60;
ft = (amp/1.0)*sin(1.0*pi*freq*simtime) + ...
     (amp/2.0)*sin(2.0*pi*freq*simtime) + ...
     (amp/6.0)*sin(3.0*pi*freq*simtime) + ...
     (amp/3.0)*sin(4.0*pi*freq*simtime);
ft = repmat(ft/1.5,nReadoutUnits,1);
ft = cell2mat(arrayfun(@(k) ft(k,:)*2^k, 1:size(ft,1),'UniformOutput',false).');

ft2 = (amp/1.0)*sin(1.0*pi*freq*simtime2) + ...
      (amp/2.0)*sin(2.0*pi*freq*simtime2) + ...
      (amp/6.0)*sin(3.0*pi*freq*simtime2) + ...
      (amp/3.0)*sin(4.0*pi*freq*simtime2);
ft2 = repmat(ft2/1.5,nReadoutUnits,1);
ft2 = cell2mat(arrayfun(@(k) ft2(k,:)*2^k, 1:size(ft,1),'UniformOutput',false).');


wo_len = zeros(nReadoutUnits,simtime_len);    
zt = zeros(nReadoutUnits,simtime_len);
zpt = zeros(nReadoutUnits,simtime_len);
x0 = 0.5*randn(N,1);
z0 = 0.5*randn(nReadoutUnits,1);

x = x0; 
r = tanh(x);
z = z0; 

figure;
ti = 0;
P = (1.0/alpha)*eye(nRec2Out);
for t = simtime
    ti = ti+1;	
    
    if mod(ti, nsecs/2) == 0
	disp(['time: ' num2str(t,3) '.']);
	subplot 211;
	plot(simtime, ft, 'linewidth', linewidth, 'color', 'green');
	hold on;
	plot(simtime, zt, 'linewidth', linewidth, 'color', 'red');
	title('training', 'fontsize', fontsize, 'fontweight', fontweight);
	legend([repmat('f',nReadoutUnits,1);repmat('z',nReadoutUnits,1)]);
	xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
	ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
	hold off;
	
	subplot 212;
	plot(simtime, wo_len, 'linewidth', linewidth);
	xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
	ylabel('|w|', 'fontsize', fontsize, 'fontweight', fontweight);
    legend(repmat('|w|',nReadoutUnits,1));
	pause(0.5);	
    end
    
    % sim, so x(t) and r(t) are created.
    x = x + (-dt*x + M*(r*dt) + wf*(z*dt))/tau;
    r = tanh(x);
    z = wo'*r;
    
    if mod(ti, learn_every) == 0
        % update inverse correlation matrix
        k = P*r;
        rPr = r'*k;
        c = 1.0/(1.0 + rPr);
        P = P - k*(k'*c);

        % update the error for the linear readout
        e = z-ft(:,ti);

        % update the output weights
        %dw = diag(-repmat(e,1,length(k))*repmat(k,1,nReadoutUnits)*c);
        dw = (-e*k.'*c).';
        wo = wo + dw;
    end
    
    % Store the output of the system.
    zt(:,ti) = z;
    wo_len(:,ti) = diag(sqrt(wo'*wo));	
end
error_avg = sum(abs(zt-ft),2)/simtime_len;
disp(['Training MAE: \n']); error_avg
disp(['Now testing... please wait.']);    


% Now test. 
ti = 0;
for t = simtime				% don't want to subtract time in indices
    ti = ti+1;    
    
    % sim, so x(t) and r(t) are created.
    x = x + (-dt*x + M*(r*dt) + wf*(z*dt))/tau;
    %x = (1.0-dt)*x + M*(r*dt) + wf*(z*dt);
    r = tanh(x);
    z = wo'*r;
    
    zpt(:,ti) = z;
end
error_avg = sum(abs(zpt-ft2),2)/simtime_len;
disp(['Testing MAE: \n']); error_avg


figure;
subplot 211;
plot(simtime, ft, 'linewidth', linewidth, 'color', 'green');
hold on;
plot(simtime, zt, 'linewidth', linewidth, 'color', 'red');
title('training', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
hold on;
ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
legend([repmat('f',nReadoutUnits,1);repmat('z',nReadoutUnits,1)]);


subplot 212;
hold on;
plot(simtime2, ft2, 'linewidth', linewidth, 'color', 'green'); 
axis tight;
plot(simtime2, zpt, 'linewidth', linewidth, 'color', 'red');
axis tight;
title('simulation', 'fontsize', fontsize, 'fontweight', fontweight);
xlabel('time', 'fontsize', fontsize, 'fontweight', fontweight);
ylabel('f and z', 'fontsize', fontsize, 'fontweight', fontweight);
legend([repmat('f',nReadoutUnits,1);repmat('z',nReadoutUnits,1)]);
	

