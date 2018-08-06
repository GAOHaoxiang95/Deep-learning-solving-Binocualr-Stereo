function y = computeDSI(row, sc, ec, f1, f2)

    
for left = sc: ec
    target = f1(row,left,:,:,:);
    for right = sc:ec
        rightPatch = f2(row, right,:,:,:);
        DSI(right, left) = ED_compareWindows(target, rightPatch)/4;
    end
      
end
y = DSI;
end