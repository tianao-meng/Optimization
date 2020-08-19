clc; clear;
close all;


% question e plot domain f0
N=5000;
x0=0;
y0=0;
R=4 * pi;
axis tight
hold on
r=R*sqrt(rand(1,N));
seta=2*pi*rand(1,N);
x=r.*cos(seta);
y=r.*sin(seta);
x=x0+x;
y=y0+y;
figure(1);
plot(x,y,'.');hold on;
axis on;
train_set = [];
train_set_ele = [];
%obtain train set

for i = 1 : N
    train_set_ele = [x(i),y(i)];
    train_set = [train_set; train_set_ele];
end


%obtain sample result
observation = [];
for i = 1 : N
    res = f0(train_set(i));
    observation = [observation; res];
end


%obtain the sample result of the f0_inf
observation_inf = [];
for i = 1 : N
    res = f0_inf(train_set(i));
    observation_inf = [observation_inf; res];
end


%normalization

%max_train_set=max(train_set);
%max_train_set

train_set = normalize (train_set);
%max_observation=max(observation);
observation = normalize (observation);


%import Neural_Network
obj = Neural_Network; 
count = [];
der_total = [];
cost_list = [];
%{
for i = 1 : 500
    i
    count = [count i];
    obj.train_gradient_decent_fix_step(train_set(i, :), observation(i));
    der_total = [der_total obj.der_total];
    cost = cost_func(obj, train_set, observation, 5000);
    cost_list = [cost_list cost];
end
%}

%train for f0_inf
for i = 1 : 100
    i
    count = [count i];
    obj.train_gradient_decent_fix_step(train_set(i, :), observation_inf(i));
    der_total = [der_total obj.der_total];
    cost = cost_func(obj, train_set, observation_inf, 5000);
    cost_list = [cost_list cost];
end


%error_y = [];
%{
for i = 1 : 50
    i
    count = [count i];
    
    output_new = obj.train_gradient_decent_backtracking(train_set(i, :), observation(i));
    der_total = [der_total obj.der_total];
    %error = (output_new{10} - observation(i)) ^ 2;
    %error_y = [error_y error];
    cost = cost_func(obj, train_set, observation, 5000);
    cost_list = [cost_list cost];

    
end
%}

%obj.square_error_y = error_y;
%{
for i = 1 : 20
    i
    count = [count i];
    obj.train_conjugate_gradient_fix_step(train_set(i, :), observation(i));
    
    cost = cost_func(obj, train_set, observation, 5000);
    cost_list = [cost_list cost];
end
%}
%{
for i = 1 : 20
    i
    count = [count i];
    obj.train_conjugate_gradient_backtracking(train_set(i, :), observation(i));
    
    cost = cost_func(obj, train_set, observation, 5000);
    cost_list = [cost_list cost];
end
%}
figure(2);
plot (count, cost_list);

%figure(3);
%plot (count, der_total);


%question f
%{
[X, Y] = meshgrid(-100:0.5:100, -100:0.5:100);
grid_res = [];
size(X)
[r,c] = size (X);
for i = 1 : r
    row_res = [];
    for j = 1 : c
        res = f0([X(i, j), Y(i,j)]);
        row_res = [row_res res];
    end
    grid_res = [grid_res;row_res];
end

ann_res = [];
for i = 1 : r
    row_res = [];
    for j = 1 : c
        
        sample = [X(i, j), Y(i,j)];
        observation = obj.Forward(sample);
        row_res = [row_res observation{10}];
    end
    ann_res = [ann_res;row_res];
end
figure(4);
plot3(X, Y, grid_res);
%contourf(X, Y, grid_res);

figure(5);
plot3(X, Y, ann_res);
%contourf(X, Y, ann_res);
%}
function cost = cost_func(obj, sample, observation, sample_num)
    cost = 0;
    for  i = 1 : sample_num
        output = obj.Forward(sample(i, :));
        cost = cost + (output{10} - observation(i))^2;
    end
    cost = cost / sample_num;
end

        
function f0_res = f0(x)
    f0_res = sin (norm(x,2)) / norm(x,2);
end

function f0_inf_res = f0_inf(x)
    %f0_inf_res = sin (norm(x,inf)) / norm(x,inf);
    f0_inf_res = norm(x,inf);
end








        
        