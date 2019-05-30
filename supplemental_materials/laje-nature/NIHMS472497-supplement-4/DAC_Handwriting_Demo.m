% DAC_Handwriting_Demo.m
% Dynamic Atractor Computing Example:
% Laje & Buonomano (2013) Nature Neuroscience
% 800 recurrent units & 2 Outputs
% Reads Trained Weights from W_Handwriting.mat
% Requires files:
% DAC_Handwriting_Demo.fig
% DAC_HandWriting_mainloop.m
% HandWritCircDiagram.jpg
% W_Handwriting.mat
% Dean Buonomano 4/10/13

function varargout = DAC_Handwriting_Demo(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @DAC_Handwriting_Demo_OpeningFcn, ...
                   'gui_OutputFcn',  @DAC_Handwriting_Demo_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

% --- Executes just before DAC_Handwriting_Demo is made visible.
function DAC_Handwriting_Demo_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

global LineHandles VoltageHandles plotXYPoint
axes(handles.axXY)
hold on
for i=1:10;
   LineHandles.plotXY(i) = plot([i/10:0.001:10*i/10],[i/10:0.001:10*i/10],'color',[i/10 i/10 i/10],'linewidth',[2]);
end
plotXYPoint = plot(10,10,'go','MarkerFaceColor','g','markersize',[6]);
 
axes(handles.axVoltage)
cmap = get(gca,'colororder');
hold on
set(gca,'ylim',[-1 11]);

for i=1:10
   VoltageHandles.plotLine(i) = plot(zeros(1,1000),'color',cmap(1+rem(i-1,7),:));
end
xlabel('Time (ms)','fontweight','bold')

axes(handles.axCircDiagram)
CircDiagram = imread('HandWritCircDiagram.jpg','jpeg');
image(CircDiagram(:,:,1:3));
axis off

% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = DAC_Handwriting_Demo_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

% --- Executes on button press in btRUN.
function btRUN_Callback(hObject, eventdata, handles)
global RUN NoiseValue PauseValue
RUN = get(gcbo,'value');
if ~RUN
   set(gcbo,'backgroundcolor',[1 0.4 0.2]);
else
   set(gcbo,'backgroundcolor',[0.8 1 0.8]);
end
drawnow;
fprintf('RUN=%d\n',RUN);

DAC_Handwriting_mainloop;

function btIN1_Callback(hObject, eventdata, handles)
global In1 InputDur
In1 = InputDur;

% --- Executes on button press in btIN2.
function btIN2_Callback(hObject, eventdata, handles)
global In2 InputDur
In2 = InputDur;

% --- Executes on button press in btPerturb.
function btPerturb_Callback(hObject, eventdata, handles)
global InPerturb InPerturbDur
InPerturb = InPerturbDur;

% --- Executes on slider movement.
function slPause_Callback(hObject, eventdata, handles)
global PauseValue
PauseValue = get(hObject,'Value');
str = sprintf('Pause = %3.2f',PauseValue);
set(handles.txtPause,'string',str);

% --- Executes on slider movement.
function slNoise_Callback(hObject, eventdata, handles)
global NoiseValue
NoiseValue = get(hObject,'Value');
str = sprintf('Noise = %5.4f',NoiseValue);
set(handles.txtNoise,'string',str);


function slPause_CreateFcn(hObject, eventdata, handles)


function slNoise_CreateFcn(hObject, eventdata, handles)
