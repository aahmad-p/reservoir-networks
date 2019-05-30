% DAC_Handwriting
% Handwriting Example: 
% Laje & Buonomano (2013) Nature Neuroscience
% called from DAC_Handwriting_Demo.m
% Dean Buonomano 4/10/13

global LineHandles VoltageHandles plotXYPoint
global InputDur In1 In2 InPerturb InPerturbDur

tau   = 10;          %10 ms (time step = 1 ms)
g=1.5;               %"gain"
InputDur = 50;       %Input Dur ms
InPerturbDur = 10;   %Pertubation Duration ms

InPerturb   = 0;
In1 = 0;
In2 = 0;
NoiseValue = get(handles.slNoise,'value');
PauseValue  = 0;
InAmp    = [0.3 2];  %[Pertub Amp, Input Amp];

%%% GRAPHICS STUFF
UpdateStep = 100;    %Updates graphics every X time steps (ms)
NumLineSegs = 10;    %Total of Segements showns
numExPlot   = 10;    %num Recurrent Units to plot
numCumTrajSteps = 2000;

%%% LOAD WEIGHT MATRICES %%%
load W_Handwriting;

%Initialize
[numEx numOut] = size(WExOut);
[numEx numIn]  = size(WInEx);
t=0;
Ex   = zeros(numEx,1);
ExV  = zeros(numEx,1);
Out  = zeros(numOut,1);
In   = zeros(numIn,1);
historyEx  = zeros(numExPlot,UpdateStep*NumLineSegs);
historyOut = zeros(numOut,UpdateStep);
historyOUT = zeros(numOut,UpdateStep,NumLineSegs);

CumulativeTraj = zeros(numCumTrajSteps,numCumTrajSteps);    %[-1:1] w/ 200 bins

%random initial state
ExV = 2*rand(numEx,1)-1;

while RUN==1
   t=t+1;
   
   %COUNT DOWN (-1) to implement the duration of the events.
   %NEGATIVE VALUES ARE NOT USED (In1>0) equals 0 or 1
   In1 = In1-1;
   In2 = In2-1;
   InPerturb = InPerturb-1;
   In = [InAmp(1)*(InPerturb>0); InAmp(2)*(In1>0); InAmp(2)*(In2>0)];

   ex_input = WExEx'*Ex + WInEx*In + NoiseValue*randn(numEx,1);   
   ExV = ExV + (-ExV + ex_input)./tau;
   Ex = tanh(ExV);
   
   out_input = WExOut'*Ex;
   Out = out_input;

   historyEx = [historyEx(:,2:end) Ex(1:10)];
   historyOut(:,rem(t-1,UpdateStep)+1) = Out;
   
   if rem(t,UpdateStep)==0
      for i=1:numExPlot
         set(VoltageHandles.plotLine(i),'xdata',[(t-UpdateStep*NumLineSegs+1):t],'ydata',historyEx(i,:)+(i-1)*1);
      end   
      for i=1:(NumLineSegs-1)
         historyOUT(:,:,i)=historyOUT(:,:,i+1);
         set(LineHandles.plotXY(i),'xdata',historyOUT(1,:,i),'ydata',historyOUT(2,:,i));
      end
      historyOUT(:,:,NumLineSegs) = historyOut;
      set(LineHandles.plotXY(NumLineSegs),'Xdata',historyOut(1,:),'Ydata',historyOut(2,:));
      set(plotXYPoint,'Xdata',historyOut(1,end),'Ydata',historyOut(2,end));
      
      fprintf('t=%5d %4.1f\n',t,tau);
      drawnow;
      assignin('base','historyEx',historyEx);
      assignin('base','historyOut',historyOut);
      assignin('base','historyOUT',historyOUT);
      assignin('base','t',t);
      assignin('base','CumulativeTraj',CumulativeTraj);
      pause(PauseValue)
            
   end

end




