clear
close all
clc

load('gtruth.mat');

%Assuming that the gtruth.mat file contains a variable named gTruth which is the output of imageLabeler app of MATLAB
image_paths = gTruth.DataSource.Source; %Original images paths
mask_paths = gTruth.LabelData.PixelLabelData; %Path of annotation images (masks) created through imageLabeler

%Total number of images
num_images = length(mask_paths);

%distances = zeros(num_images, 8); %Unused

%Create root node of XML file
docNode = com.mathworks.xml.XMLUtils.createDocument('dataset');
docRootNode = docNode.getDocumentElement;

%XML element representing the overall image set. It encloses all images in dataset
imagesElement = docNode.createElement('images');

%We are labelling 4 points (corners of the card), so there are 4 labels.
%DLib requires label IDs to be in double digit format i.e. 00, 01 etc...
partNames = {'00', '01', '02', '03'};


%Loop over all images
for i=1:num_images
    
	%Split file path into components (directory, name and extension)
	%Since we require only name and extension, we use ~ in place of directory
    [~, f, ex] = fileparts(image_paths{i});
	
	%Only the image name with extension
    image_name = strcat(f,ex);
    
	% Read annotation image and convert it to double data type. 
	% It contains only zeros and ones
    I = double(imread(mask_paths{i} ));
    
	%Get height and width of the image
	image_dims = size(I);
    disp(mask_paths{i})
	
	%Detect corners in the mask image to find out the boundary of the mask
    corners = detectMinEigenFeatures(I);
    
	%Select only the top four corners from the corner list
	%This is because only the actual 4 corners will have highest score among the detected corners
    %sortPoints (custum function written below) makes assumption that the points have been annotated in clockwise order starting from the top-left corner
	ann = sortPoints(corners.selectStrongest(4).Location);
	
	%Now ann is a 4x2 array containing 4 pixel coordinates
	
	%Create XML tag of the current image
    imageElement = docNode.createElement('image');
    imageElement.setAttribute('file',image_name);
    
	%Create XML tag for bounding box in the current image.
	%Since landmark detection assumes whole image as the region-of-interest, we specify (0,0,width,height) as the bounding box
	%IMPORTANT: Make sure that the annotations have been done on either the cropped image of the card, OR the lower face region
    boxElement = docNode.createElement('box');
    boxElement.setAttribute('top', '0');
    boxElement.setAttribute('left', '0');
    boxElement.setAttribute('width', num2str(image_dims(2))) ;
    boxElement.setAttribute('height', num2str(image_dims(1)));
    
	%Create XML tags for each annotated point
    for idx=1:4
       partElement =  docNode.createElement('part');
       partElement.setAttribute('name', partNames{idx})
       x = round(ann(idx,1));
       y = round(ann(idx,2));
       partElement.setAttribute('x', num2str(x));
       partElement.setAttribute('y', num2str(y));
       boxElement.appendChild(partElement);
    end
    
    imageElement.appendChild(boxElement);
    imagesElement.appendChild(imageElement);
    
end
docRootNode.appendChild(imagesElement);

% At this point, the XML tree has been created with the following structure:

% dataset->images->image*->box->part*

% "*" in front of the tag name indicates that there can be multiple entries of that tag e.g.
% "box" tag contains multiple "part" tags
% "images" tag contains multiple "image" tags

% Write XML file to disk
xmlFileName = 'training_with_card_landmarks.xml';
xmlwrite(xmlFileName,docNode);


%Ref: https://stackoverflow.com/a/13935419/1231073
function s = sortPoints(ann)
    x = ann(:,1);
    y = ann(:,2);
    
    cx = mean(x);
    cy = mean(y);
    
    a = atan2(y - cy, x - cx);
    
    [~, order] = sort(a);
    
    x = x(order);
    y = y(order);
    
    s = ann;
    
    s(:,1) = x;
    s(:,2) = y;
end
