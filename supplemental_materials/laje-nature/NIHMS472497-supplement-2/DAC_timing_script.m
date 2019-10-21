%% Innate Trajectory training
% Train a nonlinear random recurrent network to reproduce its own innate trajectory.
% Calls timing_kernel.m
% Written by Rodrigo Laje

% "Robust Timing and Motor Patterns by Taming Chaos in Recurrent Neural Networks"
% Rodrigo Laje & Dean V. Buonomano 2013


clear;
lwidth = 2;
fsize = 10;
set(0,'DefaultFigureWindowStyle','docked');


% parameters and hyperparameters

% network
numUnits = 800;					% number of recurrent units
numplastic_Units = 480;			% number of recurrent plastic units
p_connect = 0.1;				% sparsity parameter (probability of connection)
g = 1.5;						% synaptic strength scaling factor
scale = g/sqrt(p_connect*numUnits);	% scaling for the recurrent matrix
numInputs = 1;					% number of input units
numOut = 1;						% number of output units

% input parameters
input_pulse_value = 5.0;
start_pulse = 200;		% (ms)
reset_duration = 50;	% (ms)

% training
interval = 1000;
learn_every = 2;		% skip time points
start_train = start_pulse + reset_duration;
end_train = start_train + interval + 150;
n_learn_loops_recu = 0;		% number of training loops (recurrent)
n_learn_loops_read = 2;		% number of training loops (readout)
n_test_loops = 1;

% numerics
dt = 1;					% numerical integration time step
tmax = end_train + 200;
n_steps = fix(tmax/dt);			% number of integration time points
time_axis = [0:dt:tmax-dt];
plot_points = 500;				% max number of points to plot
plot_skip = ceil(n_steps/plot_points);
if rem(plot_skip,2)==0
	plot_skip = plot_skip + 1;
end

% firing rate model
tau = 10.0;							% time constant (ms)
sigmoid = @(x) tanh(x);			% activation function
noise_amplitude = 0;

% output function
ready_level = 0.2;
peak_level = 1;
peak_width = 30;
peak_time = start_train + interval;



% training and testing
savefile_trained = 'DAC_timing_recurr800_p0.1_g1.5.mat';

seed = RandStream('mt19937ar','seed',0);
RandStream.setGlobalStream(seed);


% create network and get innate trajectory for target
TRAIN_READOUT = 0;
TRAIN_RECURR = 0;
LOAD_DATA = 0;
GET_TARGET_INNATE_X = 1;
SAVE_DATA = 1;
n_loops = 1;
savefile = savefile_trained;
disp('getting innate activity.');

	DAC_timing_kernel;


% train recurrent
TRAIN_READOUT = 0;
TRAIN_RECURR = 1;
LOAD_DATA = 1;
loadfile = savefile_trained;
GET_TARGET_INNATE_X = 0;
SAVE_DATA = 1;
n_loops = n_learn_loops_recu;
savefile = savefile_trained;
disp('training recurrent:');

	DAC_timing_kernel;


% train readout
TRAIN_READOUT = 1;
TRAIN_RECURR = 0;
LOAD_DATA = 1;
loadfile = savefile_trained;
GET_TARGET_INNATE_X = 0;
SAVE_DATA = 1;
n_loops = n_learn_loops_read;
savefile = savefile_trained;
disp('training readout:');

	DAC_timing_kernel;


% load, run, plot
TRAIN_READOUT = 0;
TRAIN_RECURR = 0;
LOAD_DATA = 1;
loadfile = savefile_trained;
GET_TARGET_INNATE_X = 0;
SAVE_DATA = 1;
n_loops = n_test_loops;
savefile = savefile_trained;
disp('testing:');

	DAC_timing_kernel;

disp('done.');
disp('.');



%%
