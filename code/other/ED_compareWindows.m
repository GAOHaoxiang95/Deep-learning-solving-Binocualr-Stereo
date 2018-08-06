function y = ED_compareWindows(window1, window2)

sub = window1 - window2;
square = sub.*sub;
y = sum(square(:));

end
