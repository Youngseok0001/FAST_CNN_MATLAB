function patches = samplePatches(rawImages, patchSize, numPatches)

imWidth = sqrt(size(rawImages,1));
imHeight = imWidth;
numImages = size(rawImages,2);
rawImages = reshape(rawImages,imWidth,imHeight,numImages); 

patches = zeros(patchSize*patchSize, numPatches);

maxWidth = imWidth - patchSize + 1;
maxHeight = imHeight - patchSize + 1;

for num = 1:numPatches
    x = randi(maxHeight);
    y = randi(maxWidth);
    img = randi(numImages);
    p = rawImages(x:x+patchSize-1,y:y+patchSize-1, img);
    patches(:,num) = p(:);
end
    

