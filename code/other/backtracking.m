function disparity = backtracking(cost, DSI, row, col, sc, i, disparity)
oc = 1;

while row ~= sc || col ~= sc
    if (row == sc) && (col == sc)
        disparity(i, col) =col - row;
        disp('end');
    
    elseif (row == sc)
        %path(row,col) = 1;
        col = col -1;
        disparity(i, col) = col - row;
    elseif(col == sc)
        %path(row,col) = 1;
        row = row - 1;
        
    else
        if ((cost(row-1, col -1)+ DSI(row, col)) < (cost(row,col-1) + oc)) && ((cost(row-1, col -1)+ DSI(row, col)) < cost(row-1,col) + oc)
            %path(row,col) = 1;
            row = row -1;
            col = col -1;
            disparity(i, col) = col - row;
        elseif ((cost(row,col-1) + oc) < (cost(row-1, col -1)+ DSI(row, col))) && ((cost(row,col-1) + oc) < (cost(row-1, col)+oc))
            %path(row,col) = 1;
            col = col - 1;
            disparity(i, col) = col - row;
        else
            %path(row,col) = 1;
            row = row - 1;  
        end
    end
end