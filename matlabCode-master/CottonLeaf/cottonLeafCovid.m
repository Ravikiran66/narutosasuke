%imdsTrain=imageDatastore({'cl1.jpg','cl2.jpg','cl3.jpg','cl4.jpg','cl15.jpg','cl6.jpg','cl7.jpg','cl8.jpg'})
 
 


imdstrain = imageDatastore('C:\Users\bly\Downloads\matlabCode-master\CottonLeaf\shreya\train', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
imdstest = imageDatastore('C:\Users\bly\Downloads\matlabCode-master\CottonLeaf\shreya\test', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
imdsvalidation = imageDatastore('C:\Users\bly\Downloads\matlabCode-master\CottonLeaf\shreya\validation', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

%[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
     
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsvalidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
 
 
net = trainNetwork(imdstrain,layers,options);
YPred = classify(net,imdstest);
YValidation = imdstest.Labels;
 
accuracy = sum(YPred == YValidation)/numel(YValidation)
