dic = 'Baby2';
leftView = imread([dic, '\view1.png']);
rightView = imread([dic, '\view5.png']);
window_size = 9;
%parameter setting 

%imnoise(leftView,'salt & pepper', 0.02);
%imnoise(rightView,'salt & pepper', 0.02);

[R, C, d] = size(leftView);
path = zeros(R,C);

sr = 5;
sc = 5;
er = R -4;
ec = C -4;

disparity = zeros(R,C);

f1 = leftView;
f2 = rightView;
%f1 = extractAll(leftView, sr, sc, er, ec, window_size);
%f2 = extractAll(rightView, sr, sc, er, ec, window_size);

for i = sr:er
    DSI = computeDSI(i, sc, ec, f1, f2);
    %a = max(max(DSI));
    %imshow(DSI/a);
    cost = DP(DSI, sc, ec);
    disparity = backtracking(cost,DSI, ec, ec, sc, i,disparity);
    disp(i);
end
%imshow(path);
map = uint8(disparity)*3
imshow(map);
%v = RMS_error(map, 'cones/disp2.pgm');
%disp(cost);