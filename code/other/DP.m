function  y = DP(DSI, sc, ec)

oc = 1;
for left = sc: ec
    for right = sc:ec
        if (left == sc && right == sc)
            cost(right, left) = DSI(1,1);
        elseif (left == sc)
            cost(right, left) = cost(right-1,left) + oc;
        elseif (right == sc)
            cost(right, left) = cost(right,left-1) + oc;
        else
            cost(right, left) = min(min(cost(right-1, left -1) + DSI(right,left), cost(right,left-1) + oc), cost(right-1,left) + oc);
        end       
    end
end
    

y = cost;
end