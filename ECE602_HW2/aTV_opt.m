cvx_setup
%according to filename, read the grey scale image
I0 = imread('cameraman.tif');
%change f martrix element to double type. it is convient for us to
%manipulate the matrix
I0 = double(I0(33:96,81:144));

%randn('seed',2000); seed is a designed generator, 2000 means the initial number
rng(2000);

%generate 64*64 0 matrix, all masked matrix
M = false(64);

%shuffle int 1 - 64*64
ind = randperm(64*64);

%make half of matrix element available
M(ind(1:64*64/2)) = true;

%get the masked grey scale image
I = I0 .* M;
%original_res = sum(I0(:));
figure(1)
imagesc(I0)
axis square
colormap gray

figure(2)
imagesc(I)

axis square
colormap gray

n = 64;


cvx_begin
    variable I(n, n)
    minimize( aTV(I) )
    subject to
           I(ind(1:64*64/2)) == I0(ind(1:64*64/2))
cvx_end


%recovered_res_TR = sum(I(:));

gap = 0;
for i = 1 : 64
    for j = 1 : 64
        gap = gap + (I0(i, j) - I(i, j)) * (I0(i, j) - I(i, j));
    end
end
gap
figure(3)
imagesc(I)
axis square
colormap gray


function res = aTV(I)
    f = 0;
    
    for i = 2 : 64
        for j = 2 :64
            
            input1 = I(i,j) - I(i-1,j);
            input2 = I(i,j) - I(i,j -1);
            f = f + abs(input1) + abs(input2);
        end
    end
    res = f;
end