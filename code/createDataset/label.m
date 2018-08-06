file = fopen('train.txt', 'w');

N = 7003;
for i = 1:N
    fprintf(file, '%s\n', ['patches/P', num2str(i-1),'a.jpg', ' ', 'patches/P', num2str(i-1),'b.jpg']);
end

for i = 1:N
    fprintf(file, '%s\n', ['patches/N', num2str(i-1),'a.jpg', ' ', 'patches/N', num2str(i-1),'b.jpg']);
end

fclose(file);