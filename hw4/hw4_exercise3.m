clear;
cvx_setup
% center value
A_center = [60 45 -8; 
            90 30 -30; 
            0 -8 -4; 
            30 10 -10];
%radius
R = [0.05 0.05 0.05; 
     0.05 0.05 0.05; 
     0.05 0.05 0.05; 
     0.05 0.05 0.05];

b = [-6;-3;18;-9];

% get the solution A * X = B
x_ls = A_center \ b;

% get the solution of the qp formed in question 1
cvx_begin
    variables x(3) y(4) z(3)
    minimize norm(y, 2)
    subject to 
                A_center * x + R * z - b <= y;
                A_center * x - R * z - b >= -y;
                -z <= x;
                x <= z;
cvx_end 
% the optimal result of the cvx is the worst case residual of x_rls
worst_case_rls = cvx_optval;

%minimize norm( A_center *x - b, 2)
norminal_ls = norm(A_center * x_ls - b, 2);
norminal_rls = norm(A_center * x - b, 2);

% get the worst case residual of x_ls
r = A_center * x_ls - b;
delta = ones(4,3);
c = 0.05;
% sign return 1 0 -1
for i=1:length(r)
    if r(i) < 0
        delta(i,:) = -c * sign(x_ls');
    else
        delta(i,:) = c * sign(x_ls');
    end
end
worst_case_ls = norm(r + delta * x_ls, 2);

