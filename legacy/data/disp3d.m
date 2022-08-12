function [ output_args ] = disp3d( myimageIn, myimage_title, figure_control, image_type )

% -----------------------------------------------------------
% -------------------------- ABOUT --------------------------
% disp3d was made in July 2011 by Danny Park and Hyrum Griffin.
% It is a stand-alone function that will display a 3D dataset
% with a variety of gui tools to facilitate viewing.
%
% -----------------------------------------------------------
% ------------------------- INPUTS --------------------------
%
% The inputs must be in this order:
%  disp3d( myimageIn, myimage_title, figure_control, image_type )
%
% myimageIn                                     required
% -- This is the 3D dataset itself.
%
% myimage_title                                 optional
% -- If specified this string will be displayed both above the
% -- image and in the figure title. 
%
% figure_control                                optional
% -- This option is similar to the "figure()" command.  
% -- There are 2 allowed values, 'new' and 'replace'.
% -- 'new' is the default value and will cause the image
% -- to appear in a new figure.  'replace' will search for
% -- an existing figure with the exact same title, and if
% -- it finds one it will replace that figure with the new
% -- one.  If 'replace' cannot find an existing figure it
% -- will behave like 'new'.
%
% image_type                                    optional
% -- Preloads which component of the image is displayed.
% -- The 4 allowed values are 'Magnitude', 'Phase', 'Real',
% -- and 'Imaginary'.  If not specified, the viewer will
% -- default to 'Magnitude'.
%

% -----------------------------------------------------------
% -------------- INPUT ARG PARSING & VERIFYING --------------

if nargin < 1,
    error('No image to display');
end

if nargin < 2,
    myimage_title = 'No title specified';
end

if nargin < 3,
    figure_control = 'new';
end

if nargin < 4,
    image_type = 'Magnitude';
end
ChangeType(image_type);

if ~(strcmp(figure_control,'new') || strcmp(figure_control,'replace'))
    figure_control = 'new';
end

if ndims(myimage)~=3,
    error('myimage needs to be a 3D image');
end

% -----------------------------------------------------------
% --------------------- ADDITIONAL ARGS ---------------------

% BACKGROUND COLOR
% bgcolor will set the background color of the window.  Colors
% are specified in rgb values ranging from 0 to 1.
bgcolor = [.8 .8 .8]; % [r g b] -> [.8 .8 .8] are default values

% SLIDER PADDING
% windowingSliderPadding is used to specify extra space to be
% given on either side of the windowing sliders.  For example, 
% if [0 0] is specified then there will be no padding.  If 
% [.25 .1] is specified then an additional 25% of the range
% will be padded on the left, and 10% on the right.
windowingSliderPadding = [.25 .25]; % [padOnLeft padOnRight]


% -----------------------------------------------------------
% ------------------------ CODE BODY ------------------------


mysize = size(myimage);
slice_dim = 3;
cur_slice = round(mysize(slice_dim)/2);
pmax = max(myimage(:));
pmin = min(myimage(:));
prev_click = 0;
check = 0;

        % [right,up,wide,high]
        

if strcmp(figure_control,'new')        
    close_old = 0;
elseif strcmp(figure_control,'replace')
    if ishandle(findobj('type','figure','name',['Display 3D Data: ' myimage_title]))
        close_old = 1;
    else
        close_old = 0;
    end
else
    close_old = 0;
end




a = @click_callback; % Create a function handle to the click_callback function
% Create the window
f = figure('Visible','off',...
    'NumberTitle','off',...
    'Position',[0,0,600,600],...
    'ButtonDownFcn',a,...
    'Color',bgcolor,...
    'Toolbar','figure');
    
    
% Coordinates of the image
axs_draw = axes('Units','pixels',...
        'Position',[25,50,350,500]);
        
    
    
    
    
    
% variables for slice selection position
sliceControlX = 420;
sliceControlY = 535;
        
% Text saying Slice Selection
Slice_Text = uicontrol('Style','text',...
        'FontSize',14,...
        'BackgroundColor',bgcolor,...
        'Position',[sliceControlX,sliceControlY,150,20],...
        'String','Slice Selection');
    
% Text saying image dimensions
Dimensions = uicontrol('Style','text',...
        'BackgroundColor',bgcolor,...
        'Position',[sliceControlX+37,sliceControlY-30,75,15],...
        'String',num2str(mysize));
 
% Dim1 Button
dim1btn = uicontrol('Style','pushbutton',...
             'String',':',...
             'Position',[sliceControlX+37,sliceControlY-30-30,25,20],...
             'Callback',{@dim1btn_press});
% Dim2 Button
dim2btn = uicontrol('Style','pushbutton',...
             'String',':',...
             'Position',[sliceControlX+37+25,sliceControlY-30-30,25,20],...
             'Callback',{@dim2btn_press});
% Dim3 Button
dim3btn = uicontrol('Style','pushbutton',...
             'String',':',...
             'Position',[sliceControlX+37+50,sliceControlY-30-30,25,20],...
             'Callback',{@dim3btn_press});

  
% Slider used for slice selection
slider = uicontrol('Style', 'slider',...
        'Min',1,'Max',mysize(slice_dim),'Value',round(mysize(slice_dim)/2),...
        'SliderStep',[1/mysize(slice_dim) 1/20],...
        'Position', [sliceControlX,sliceControlY-30-60,150,20],...
        'Callback', {@slider_action}); 
    
% Checkbox
checkbox = uicontrol('Style','checkbox',...
             'Min',0,'Max',1,...
             'String',['Rotate View CCW 90' char(176)],...
             'Value', 0,...
             'BackgroundColor',bgcolor,...
             'Position',[sliceControlX+10,sliceControlY-30-60-30,149,20],...
             'Callback',{@checkbox_click});
    
    
    
    
% variables for slice selection position
windowControlX = sliceControlX;
windowControlY = 240;
    
        
% Text saying Windowing
Windowing_Text = uicontrol('Style','text',...
        'FontSize',14,...
        'BackgroundColor',bgcolor,...
        'Position',[windowControlX,windowControlY,150,25],...
        'String','Windowing');
    
% Text saying image max
ImMax = uicontrol('Style','text',...
        'BackgroundColor',bgcolor,...
        'Position',[windowControlX-25,windowControlY-30,200,15],...
        'String',['Image Max = ' num2str(max(myimage(:)))]);
 
% Text saying image min
ImMin = uicontrol('Style','text',...
        'BackgroundColor',bgcolor,...
        'Position',[windowControlX-25,windowControlY-30-15,200,15],...
        'String',['Image Min = ' num2str(min(myimage(:)))]);

sliderMin = -100;
sliderMax =  100;
sliderMinMap = pmin - (pmax-pmin)*windowingSliderPadding(1);
sliderMaxMap = pmax + (pmax-pmin)*windowingSliderPadding(2);
  
% Text saying window max
WinMax = uicontrol('Style','text',...
        'BackgroundColor',bgcolor,...
        'Position',[windowControlX-25,windowControlY-30-15-30,200,15],...
        'String',['Window Max = ' num2str(pmax)]);

% Slider used for setting max window
WinMaxSlider = uicontrol('Style', 'slider',...
        'Min',sliderMin,'Max',sliderMax,'Value',sliderMin+(sliderMax-sliderMin)*(windowingSliderPadding(1)+1)/(windowingSliderPadding(1)+1+windowingSliderPadding(2)),...
        'SliderStep',[1/(sliderMax-sliderMin+1) 1/20],...
        'Position',[windowControlX,windowControlY-30-15-30-20,150,20],...
        'Callback', {@MaxSlider}); 
 
% Text saying window min
WinMin = uicontrol('Style','text',...
        'BackgroundColor',bgcolor,...
        'Position',[windowControlX-25,windowControlY-30-15-25-20-25,200,15],...
        'String',['Window Min = ' num2str(pmin)]);
    
% Slider used for setting min window
WinMinSlider = uicontrol('Style', 'slider',...
        'Min',sliderMin,'Max',sliderMax,'Value',sliderMin+(sliderMax-sliderMin)*(windowingSliderPadding(1))/(windowingSliderPadding(1)+1+windowingSliderPadding(2)),...
        'SliderStep',[1/(sliderMax-sliderMin+1) 1/20],...
        'Position',[windowControlX,windowControlY-30-15-25-20-25-20,150,20],...
        'Callback', {@MinSlider}); 
 
% Reset Windowing Button
resetWindowing = uicontrol('Style','pushbutton',...
             'String','Reset Windowing',...
             'Position',[windowControlX+25,windowControlY-30-15-25-20-25-20-40,100,20],...
             'Callback',{@window_reset});
    

    
    % [right,up,wide,high]
    
% Radio Buttons

% Create the button group.
h = uibuttongroup('visible','off','Position',[.75 .52 .15 .15],'BackgroundColor',bgcolor);
% Create radio buttons in the button group.
r1 = uicontrol('Style','Radio','String','Magnitude','BackgroundColor',bgcolor,...
    'pos',[5 65 80 20],'parent',h,'HandleVisibility','off');
r2 = uicontrol('Style','Radio','String','Phase','BackgroundColor',bgcolor,...
    'pos',[5 45 80 20],'parent',h,'HandleVisibility','off');
r3 = uicontrol('Style','Radio','String','Real','BackgroundColor',bgcolor,...
    'pos',[5 25 80 20],'parent',h,'HandleVisibility','off');
r4 = uicontrol('Style','Radio','String','Imaginary','BackgroundColor',bgcolor,...
    'pos',[5 5 80 20],'parent',h,'HandleVisibility','off');
% Initialize some button group properties. 
set(h,'SelectionChangeFcn',@selcbk);
set(h,'Visible','on');

switch image_type
    case 'Magnitude'
        set(h,'SelectedObject',[r1]);  
    case 'Phase'
        set(h,'SelectedObject',[r2]); 
    case 'Real'
        set(h,'SelectedObject',[r3]); 
    case 'Imaginary'
        set(h,'SelectedObject',[r4]); 
end
        
function selcbk(source,eventdata)
    ChangeType(get(eventdata.NewValue,'String'));
    pmax = max(myimage(:));
    pmin = min(myimage(:));
    set(ImMin,'String',['Image Min = ' num2str(min(myimage(:)))]);
    set(ImMax,'String',['Image Max = ' num2str(max(myimage(:)))]);
    sliderMinMap = pmin - (pmax-pmin)*windowingSliderPadding(1);
    sliderMaxMap = pmax + (pmax-pmin)*windowingSliderPadding(2);
    set(WinMaxSlider,'Value',sliderMin+(sliderMax-sliderMin)*(windowingSliderPadding(1)+1)/(windowingSliderPadding(1)+1+windowingSliderPadding(2)));
    set(WinMinSlider,'Value',sliderMin+(sliderMax-sliderMin)*(windowingSliderPadding(1))/(windowingSliderPadding(1)+1+windowingSliderPadding(2)));
    display_all();
end

    
% vname = @(x) inputname(1);
% image_title = vname(myimage)



% Change units to normalized so components resize properly
set([h,r1,r2,r3,r4,resetWindowing,axs_draw,Slice_Text,Dimensions,dim1btn,dim2btn,dim3btn,slider,checkbox,Windowing_Text,ImMax,ImMin,WinMax,WinMaxSlider,WinMin,WinMinSlider],'Units','normalized');



% center gui
movegui(f,'center');

display_all();

% make gui visible
set(f,'Visible','on');




    function display_all()
        
        if close_old == 1
            pos = get(findobj('type','figure','name',['Display 3D Data: ' myimage_title]),'Position');
            if iscell(pos)
                pos = cell2mat(pos(1));
            end
            set(f,'Position',pos);
            close(findobj('type','figure','name',['Display 3D Data: ' myimage_title]));
            close_old = 0;
        end
        
        set(f,'Name',['Display 3D Data: ' myimage_title]);
        
        switch slice_dim,
            case 1
                if check
                imshow(squeeze(myimage(cur_slice,:,size(myimage,3):-1:1))',[pmin pmax]);
                else
                imshow(squeeze(myimage(cur_slice,:,:)),[pmin pmax]);   
                end
                set(dim1btn,'String',num2str(cur_slice));
                set(dim2btn,'String',':');
                set(dim3btn,'String',':');
            case 2
                if check
                imshow(squeeze(myimage(:,cur_slice,size(myimage,3):-1:1))',[pmin pmax]);
                else
                imshow(squeeze(myimage(:,cur_slice,:)),[pmin pmax]);    
                end
                set(dim1btn,'String',':');
                set(dim2btn,'String',num2str(cur_slice));
                set(dim3btn,'String',':');
            case 3
                if check
                imshow(squeeze(myimage(:,size(myimage,2):-1:1,cur_slice))',[pmin pmax]);
                else
                imshow(squeeze(myimage(:,:,cur_slice)),[pmin pmax]);    
                end
                set(dim1btn,'String',':');
                set(dim2btn,'String',':');
                set(dim3btn,'String',num2str(cur_slice));
        end
        set(slider,'Value',cur_slice);
        set(WinMaxSlider,'Value',(pmax - sliderMinMap)/(sliderMaxMap-sliderMinMap)*(sliderMax-sliderMin)+sliderMin);
        set(WinMinSlider,'Value',(pmin - sliderMinMap)/(sliderMaxMap-sliderMinMap)*(sliderMax-sliderMin)+sliderMin);
        set(WinMax,'String',['Window Max = ' num2str(pmax,4)]);
        set(WinMin,'String',['Window Min = ' num2str(pmin,4)]);
        title(myimage_title);
        titlehandle = get(gca, 'title');
        set(titlehandle, 'FontSize', 11)
        
    end

    function ChangeType(Type)
        switch Type
            case 'Magnitude'
                myimage = abs(myimageIn); 
            case 'Phase'
                myimage = unwrap(angle(myimageIn));
            case 'Real'
                myimage = real(myimageIn);
            case 'Imaginary'
                myimage = imag(myimageIn);
        end
    end

    function dim1btn_press(source,eventdata)
        str = get(source,'String');
        if str == ':'
            slice_dim = 1;
            cur_slice = round(mysize(slice_dim)/2);
            set(slider,'Max',mysize(slice_dim));  
            set(slider,'SliderStep',[1/mysize(slice_dim) 1/20]);
        end
        display_all();
    end
    function dim2btn_press(source,eventdata)
        str = get(source,'String');
        if str == ':'
            slice_dim = 2;
            cur_slice = round(mysize(slice_dim)/2);
            set(slider,'Max',mysize(slice_dim));  
            set(slider,'SliderStep',[1/mysize(slice_dim) 1/20]);
        end
        display_all();
    end
    function dim3btn_press(source,eventdata)
        str = get(source,'String');
        if str == ':'
            slice_dim = 3;
            cur_slice = round(mysize(slice_dim)/2);
            set(slider,'Max',mysize(slice_dim));  
            set(slider,'SliderStep',[1/mysize(slice_dim) 1/20]);
        end
        display_all();
    end

    function slider_action(source,eventdata) 
        cur_slice = round(get(source,'Value'));
        display_all();
    end
    function checkbox_click(source,eventdata)
        check = get(source,'Value');
        display_all();
    end


    function click_callback(source,eventdata)
         if strcmp(get(source,'SelectionType'),'normal')
            %left click
            prev_click = 0;
            cur_slice = cur_slice - 1;
            if cur_slice < 1,
                cur_slice = 1;
            end
         elseif strcmp(get(source,'SelectionType'),'alt')
            %right click
            prev_click = 1;
            cur_slice = cur_slice + 1;
            if cur_slice > size(myimage,slice_dim),
                cur_slice = size(myimage,slice_dim);
            end
         elseif strcmp(get(source,'SelectionType'),'open')
             if prev_click == 0
                 %left click
                cur_slice = cur_slice - 1;
                if cur_slice < 1,
                    cur_slice = 1;
                end
             else
                 %right click
                cur_slice = cur_slice + 1;
                if cur_slice > size(myimage,slice_dim),
                    cur_slice = size(myimage,slice_dim);
                end
             end
         end
        display_all();
    end

    function MaxSlider(source,eventdata) 
        pmax = (get(source,'Value') - sliderMin)/(sliderMax-sliderMin)*(sliderMaxMap-sliderMinMap)+sliderMinMap;
        if pmax > sliderMaxMap
            pmax = sliderMaxMap;
        end
        if pmax <= pmin
            pmax = pmin + (sliderMaxMap-sliderMinMap)/(sliderMax-sliderMin);
        end
        display_all();
    end
    function MinSlider(source,eventdata) 
        pmin = (get(source,'Value') - sliderMin)/(sliderMax-sliderMin)*(sliderMaxMap-sliderMinMap)+sliderMinMap;
        if pmin < sliderMinMap
            pmin = sliderMinMap;
        end
        if pmin >= pmax
            pmin = pmax - (sliderMaxMap-sliderMinMap)/(sliderMax-sliderMin);
        end
        display_all();
    end

    function window_reset(source,eventdata)
        pmax = max(myimage(:));
        pmin = min(myimage(:));
        display_all();
    end


end

