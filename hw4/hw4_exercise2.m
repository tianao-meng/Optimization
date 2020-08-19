clear;
cvx_setup
m=256;
t=linspace(0,1,m)';
y=exp(-128*((t-0.3).^2))-3*(abs(t-0.7).^0.4);
mpdict=wmpdictionary(m,'LstCpt',{'dct',{'wpsym4',2}});
A=full(mpdict);

cvx_begin
    variables x1(1, 512) t_l1_norm(1, 512)
    minimize t_l1_norm * ones(512,1)
    subject to 
                x1 >= -t_l1_norm;
                x1 <= t_l1_norm;
                A * transpose(x1) == y;
    
cvx_end

cvx_begin
    variables x2(1, 512)
    minimize sum(x2.^2)
    subject to 
                A * transpose(x2) == y;
    
cvx_end
    
res1 = A * transpose(x1) - y;
res2 = A * transpose(x2) - y;
relative_error1 = sum(res1.^2) / sum(y.^2);
relative_error2 = sum(res2.^2) / sum(y.^2);

x1_5 = max_entries(0.05, x1);
[x1_5_error, rescon1_5] = cal(x1_5, y, A);
x2_5 = max_entries(0.05, x2);
[x2_5_error, rescon2_5] = cal(x2_5, y, A);

x1_5_accuarcy = 1 - x1_5_error;
x2_5_accuarcy = 1 - x2_5_error;

x1_3 = max_entries(0.03, x1);
[x1_3_error, rescon1_3] = cal(x1_3, y, A);
x2_3 = max_entries(0.03, x2);
[x2_3_error, rescon2_3] = cal(x2_3, y, A);

x1_1 = max_entries(0.01, x1);
[x1_1_error, rescon1_1] = cal(x1_1, y, A);
x2_1 = max_entries(0.01, x2);
[x2_1_error, rescon2_1] = cal(x2_1, y, A);

x2_percentage = max_entries(0.181, x2);
[x2_percentage_error, rescon2_percentage] = cal(x2_percentage, y, A);
%{
cvx_begin
    variables percentage

    minimize (cal_error(x1_3_error, percentage, x2, y, A))
    
    subject to
            percentage >= 0;
    
cvx_end
%}
%{
syms percentage;
percentage_res = double(solve(cal_error(x1_3_error, percentage, x2, y, A)));
%}
figure(1)
plot(t, y)
hold on
plot(t, rescon1_5)
hold on 
plot(t, rescon2_5)
legend('y','reconstruct l1 norm', 'reconstruct l2 norm')
title('5%')

figure(2)
plot(t, y)
hold on
plot(t, rescon1_3)
hold on 
plot(t, rescon2_3)
legend('y','reconstruct l1 norm', 'reconstruct l2 norm')
title('3%')

figure(3)
plot(t, y)
hold on
plot(t, rescon1_1)
hold on 
plot(t, rescon2_1)
legend('y','reconstruct l1 norm', 'reconstruct l2 norm')
title('1%')


function x_res = max_entries(percentage, x)
    
    x_mid = x;
    x_res = x;
    x_abs = abs(x_mid);
    num_keep_5 = 512 * percentage;
    [max,max_index_list]=sort(x_abs(:));
    x_res(max_index_list(1 : end - num_keep_5)) = 0;
   

    
end

function [error, res_recon] = cal(x, y, A)
    res = A * transpose(x) - y;
    error = sum(res.^2) / sum(y.^2);
    res_recon = A * transpose(x);
end

function error = cal_error(x1_3_error, percentage, x, y, A)
   
    x_mid = x;
    x_res = x;
    x_abs = abs(x_mid);
    num_keep_5 = 512 * percentage;
    
    [max,max_index_list]=sort(x_abs(:));
    x_res(max_index_list(1 : end - num_keep_5)) = 0;
    res = A * transpose(x_res) - y;
    error = x1_3_error - sum(res.^2) / sum(y.^2);
    
    
end


