%% Innate Trajectory training
% Called by DAC_timing_script.m
% Written by Rodrigo Laje

% "Robust Timing and Motor Patterns by Taming Chaos in Recurrent Neural Networks"
% Rodrigo Laje & Dean V. Buonomano 2013


plot_recurrent_activity = 0;

if GET_TARGET_INNATE_X == 1
	noise_amp = 0;
else
	noise_amp = noise_amplitude;
end

if LOAD_DATA == 1
	load(loadfile);
else
%% connectivity matrices

	% random sparse recurrent matrix between units.
	% indices in WXX are defined as WXX(postyn,presyn),
	% that is WXX(i,j) = connection from X(j) onto X(i)
	% then the current into the postsynaptic unit is simply
	% (post)X_current = WXX*(pre)X.

	% if p_connect is very small, you can use WXX = sprandn(numUnits,numUnits,p_connect)*scale;
	% otherwise, use the following ("sprandn will generate significantly fewer nonzeros than requested if m*n is small or density is large")
	WXX_mask = rand(numUnits,numUnits);
	WXX_mask(WXX_mask <= p_connect) = 1;
	WXX_mask(WXX_mask < 1) = 0;
	WXX = randn(numUnits,numUnits)*scale;
	WXX = sparse(WXX.*WXX_mask);
	WXX(logical(eye(size(WXX)))) = 0;	% set self-connections to zero
	WXX_ini = WXX;

	% input connections WInputX(postsyn,presyn)
	WInputX = 1*randn(numUnits,numInputs);

	% output connections WXOut(postsyn,presyn)
	WXOut = randn(numOut,numUnits)/sqrt(numUnits);
	WXOut_ini = WXOut;


%% input

	start_pulse_n = round(start_pulse/dt);
	reset_duration_n = round(reset_duration/dt);
	start_train_n = round(start_train/dt);
	end_train_n = round(end_train/dt);

	input_pattern = zeros(numInputs,n_steps);
	input_pattern(1,start_pulse_n:start_pulse_n+reset_duration_n - 1) = input_pulse_value*ones(1,reset_duration_n);


%% output target

	bell = normaldistribution(time_axis,peak_time,peak_width);
	bell_max = max(bell);
	target_Out = ready_level + ((peak_level-ready_level)/bell_max)*bell;


end


%% P matrix definition

if (TRAIN_RECURR == 1)
	plastic_units = [1:numplastic_Units];		% list of all recurrent units subject to plasticity
	% one P matrix for each plastic unit in the network, plus a P matrix for the readout
	delta = 1.0;				% RLS: P matrix initialization
	for i = 1:numplastic_Units;
		pre_plastic_units(i).inds = find(WXX(plastic_units(i),:));	% list of all units presynaptic to plastic_units
		num_pre_plastic_units(i) = length(pre_plastic_units(i).inds);
		P_recurr(i).P = (1.0/delta)*eye(num_pre_plastic_units(i));
	end
end
if (TRAIN_READOUT == 1)
	delta = 1.0;				% RLS: P matrix initialization
	P_readout = (1.0/delta)*eye(numUnits);
end



%% main loop

%figure(1);
%clf(1);


X_history = zeros(numUnits,n_steps);
Out_history = zeros(numOut,n_steps);

% training/testing loop
for j = 1:n_loops
    figure;   
	fprintf('  loop: ');

	% auxiliary variables for the training plot
	WXOut_len = zeros(1,n_steps);
	WXX_len = zeros(1,n_steps);
	dW_readout_len = zeros(1,n_steps);
	dW_recurr_len = zeros(1,n_steps);
	train_window = 0;

	% initial conditions: try - nonrandom initial conditions
	%Xv = 1*(2*rand(numUnits,1)-1);
    Xv = zeros(numUnits,1);
	X = sigmoid(Xv);    %tanh
	Out = zeros(numOut,1);


	% integration loop
	for i = 1:n_steps

		if rem(i,round(n_steps/10)) == 0 && (TRAIN_RECURR == 1 || TRAIN_READOUT == 1)
			fprintf('.');
		end

		Input = input_pattern(:,i);

		% update units: X is the firing rate
		noise = noise_amp*randn(numUnits,1)*sqrt(dt);
		Xv_current = WXX*X + WInputX*Input + noise;
		Xv = Xv + ((-Xv + Xv_current)./tau)*dt;
		X = sigmoid(Xv);
		Out = WXOut*X;

		% start-end training window
		if (i == start_train_n)
			train_window = 1;
		end
		if (i == end_train_n)
			train_window = 0;
		end

		% training
		if (train_window == 1 && rem(i,learn_every) == 0)

			if TRAIN_RECURR == 1
				% train recurrent
				error = X - Target_innate_X(:,i);
				for plas = 1:numplastic_Units
					X_pre_plastic = X(pre_plastic_units(plas).inds);
					P_recurr_old = P_recurr(plas).P;
					P_recurr_old_X = P_recurr_old*X_pre_plastic;
					den_recurr = 1 + X_pre_plastic'*P_recurr_old_X;
					P_recurr(plas).P = P_recurr_old - (P_recurr_old_X*P_recurr_old_X')/den_recurr;
					% update network matrix
					dW_recurr = -error(plas)*(P_recurr_old_X/den_recurr)';
					WXX(plas,pre_plastic_units(plas).inds) = WXX(plas,pre_plastic_units(plas).inds) + dW_recurr;
					% store change in weights
					dW_recurr_len(i) = dW_recurr_len(i) + sqrt(dW_recurr*dW_recurr');
				end
			end

			if TRAIN_READOUT == 1
				% update inverse correlation matrix (using property P' = P)
				P_readout_old = P_readout;
				P_readout_old_X = P_readout_old*X;
				den_readout = 1 + X'*P_readout_old_X;
				P_readout = P_readout_old - (P_readout_old_X*P_readout_old_X')/den_readout;
				% update error
				error = Out - target_Out(i);
				% update output weights
				dW_readout = -error*(P_readout_old_X/den_readout)';
				WXOut = WXOut + dW_readout;
				% store change in weights
				dW_readout_len(i) = sqrt(dW_readout*dW_readout');
			end

		end
		% store output
		Out_history(:,i) = Out;
		X_history(:,i) = X;
		WXOut_len(i) = sqrt(sum(reshape(WXOut.^2,numOut*numUnits,1)));
		WXX_len(i) = sqrt(sum(reshape(WXX.^2,numUnits^2,1)));
	end

	fprintf(' %2d/%2d\n',j,n_loops);

	% plot
	% input, output, target
	if(plot_recurrent_activity==1)
        subplot(4,1,1);
    else
        subplot(2,1,1);
    end
	plot(time_axis(1:plot_skip:length(target_Out))-start_train, target_Out(1:plot_skip:end),'g-','linewidth', lwidth);
	hold on;
	for input_nbr = 1:numInputs
		plot(time_axis(1:plot_skip:end)-start_train, 0.5*input_pattern(input_nbr,1:plot_skip:end),'b-','linewidth', lwidth);
	end
	plot(time_axis(1:plot_skip:end)-start_train, squeeze(Out_history(1,1:plot_skip:i)), 'r-','linewidth', lwidth);
	ylabel('Input/2, output, target', 'fontsize', fsize);
	xlim([time_axis([1 end]) - start_train]);

	% recurrent activity
	if(plot_recurrent_activity==1)
        subplot(4,1,[2 3]);
        for x_unit = 1:10
            plot(time_axis(1:plot_skip:end)-start_train, X_history(x_unit,1:plot_skip:end)+2*x_unit);
            hold all;
        end
        hold on
        xlim([time_axis([1 end]) - start_train]);
        ylabel('Recurrent units', 'fontsize', fsize);
    end

	% training measures
	if(plot_recurrent_activity==1)
        subplot(4,1,4);
    else
        subplot(2,1,2);
    end
	if TRAIN_RECURR == 0
		plot(time_axis(1:plot_skip:end)-start_train, WXOut_len(1:plot_skip:end), 'linewidth', lwidth);
		ylabel('|WXOut|', 'fontsize', fsize);
		legend('|WXOut|','Location','SouthEast');
	else
		plot(time_axis(1:plot_skip:end)-start_train, WXX_len(1:plot_skip:end), 'linewidth', lwidth);
		ylabel('|WXX|', 'fontsize', fsize);
		legend('|WXX|','Location','SouthEast');
	end
	xlim([time_axis([1 end]) - start_train]);
	xlabel('time (ms)');
	pause(0.5);
    
    %mtit
    title({sprintf('GET INNATE TRAJECTORY: %d | RECURRENT: %d',GET_TARGET_INNATE_X, TRAIN_RECURR),...
        sprintf('READOUT: %d | TESTING: %d | loop: %2d/%2d',TRAIN_READOUT, ~(TRAIN_RECURR&TRAIN_READOUT&GET_TARGET_INNATE_X), j,n_loops)});
    pause(0.1);

end


% get target from innate trajectory
if GET_TARGET_INNATE_X == 1
	Target_innate_X = X_history;
elseif ~exist('Target_innate_X','var')
	Target_innate_X = [];
end


if SAVE_DATA == 1
	save(savefile,'WXX','WXX_ini','WInputX','WXOut','WXOut_ini','Target_innate_X','target_Out',...
		'numUnits','numplastic_Units','p_connect','g','numInputs','numOut',...
		'plot_skip','learn_every','tau','sigmoid','scale','noise_amplitude',...
		'input_pulse_value','start_pulse','reset_duration','input_pattern',...
		'interval','learn_every','start_train','end_train',...
		'n_learn_loops_recu','n_learn_loops_read','n_test_loops',...
		'dt','tmax','n_steps','time_axis','plot_points','plot_skip',...
		'tau','sigmoid','noise_amplitude');
end


%%
