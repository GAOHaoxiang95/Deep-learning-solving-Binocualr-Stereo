dic = 'Moebius';

im = imread([dic, '\view1.png']);
im2 = imread([dic, '\view5.png']);
d = imread([dic, '\disp1.png']);

fold = 'test\';
counter = 0;
scale = 3;

[R, C] = size(d);
Opos = [-1, 0, 1];
%d = zeros(R,C);

for i = 9:R
    for j = 9:C
        index = randi(3);
        %d(i,j) = d(i,j) + Opos(index);
    end
end

d = int32(d/scale);%scaled disparity

for i = 50: 9: R
    for j = 5: 9: C
        random = 20 + randi(20);%random value, used in negative class
        flag = randi(2);
        if flag == 2
            random = -random;
        end
        %disp(random);
        if ((d(i,j)~= 0) && (j < C-4) && (i < R-4) && (i > 4) && (j > 4) && (j-d(i,j) > 5) && (j-d(i,j)-random > 5) && (j-d(i,j)-random +5 < C))
            x1 = im(i,j,1);
            y1 = im(i,j,2);
            z1 = im(i,j,3);
            
            x2 = im2(i,j-d(i,j),1);
            y2 = im2(i,j-d(i,j),2);
            z2 = im2(i,j-d(i,j),3);
            if (z1 - z2 < 10 && z2 - z1 < 10 && x1 - x2 < 10 && x2 - x1 < 10 && y1 - y2 < 10 && y2 - y1 < 10)
               
                name1 = [fold, 'P', num2str(counter), 'a.jpg'];
                name2 = [fold, 'P', num2str(counter), 'b.jpg'];
                extractPatches(i, j, im, im2, name1, name2, d(i,j));
              
                name1 = [fold, 'N', num2str(counter), 'a.jpg'];
                name2 = [fold, 'N', num2str(counter), 'b.jpg'];
                extractPatches(i, j, im, im2, name1, name2, d(i,j)+ random);
                counter = counter + 1;
            end
        end
    end
end
disp(counter);   
        