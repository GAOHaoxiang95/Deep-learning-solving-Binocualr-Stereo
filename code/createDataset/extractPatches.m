function out = extractPatches(i, j, im1, im2, name1, name2, d)
%this function is used to extract and save image patches, called by
%createDatasets transcript
patch1 = zeros(9,9,3);
patch2 = zeros(9,9,3);

x = 1;
y = 1;
for a = (i-4):(i+4)
    for b = (j-4):(j+4)
        patch1(x,y,1) = im1(a,b,1);
        patch1(x,y,2) = im1(a,b,2);
        patch1(x,y,3) = im1(a,b,3);
        patch2(x,y,1) = im2(a,b-d,1);
        patch2(x,y,2) = im2(a,b-d,2);
        patch2(x,y,3) = im2(a,b-d,3);
       
        y = y + 1;
    end
    y = 1;
    x = x+1;
end
patch1 = uint8(patch1);
patch2 = uint8(patch2);

%patch1 = imrotate(patch1, 180, 'nearest', 'loose');%data augmentation term
%patch2 = imrotate(patch2, 180, 'nearest', 'loose');
imwrite(patch1,name1);
imwrite(patch2,name2);
out = 0;
end