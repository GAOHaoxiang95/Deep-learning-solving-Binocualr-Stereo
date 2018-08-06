function s = RMS_error(test, groundTruth)
t = double(test);
g = double(imread(groundTruth));

[R, C, d] = size(g);
%disp(d);
crop = g(5:R-4, 5:C-4);
%t = t(5:R-5, 5:C-5);
crop = crop/255;
t = t/255;

sub = crop - t;
%disp(sub);
absolute = abs(sub);

s = sum(absolute(:));
%disp(s);
s = sqrt(s/((R-8)*(C-8)));

end

