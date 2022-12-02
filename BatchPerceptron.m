load('hw4.mat')
class1 = hw4_2_1;
class2 = hw4_2_2;

c1_x1 = class1(1,:);
c1_x2 = class1(2,:);
plot(c1_x1, c1_x2, '.')
figure()
c1_x3 = c1_x1 .* c1_x2;
c1_xnew = [ones(1,100); c1_x1; c1_x2; c1_x3];
plot3(c1_x1, c1_x2, c1_x3, '.')
figure()

c2_x1 = class2(1,:);
c2_x2 = class2(2,:);
plot(c2_x1, c2_x2, '.')
figure()
c2_x3 = c2_x1 .* c2_x2;
c2_xnew = [ones(1,100); c2_x1; c2_x2; c2_x3];
plot3(c2_x1, c2_x2, c2_x3, '.')

%% Part 2
c2_x1_neg = -1 .* (c2_x1);
c2_x2_neg = -1 .* (c2_x2);
eta = 1;                    % learning rate
theta = 1;                  % threshold
c2_xnew_neg = [ones(1,100); c2_x1_neg; c2_x2_neg; c2_x3];    % negative class2
c_xnew = [c1_xnew, c2_xnew_neg];
c_xnew = normc(c_xnew);
crit_fun = @(a, y) a(:,1) + a(:,2:4) * y(2:4,:);   % discriminant function
a_vec = abs(sum(c_xnew, 2));
a_vec = normc(a_vec);
thresh = 10;

while(thresh > theta)
    vec = normr(crit_fun(transpose(a_vec), c_xnew));
    ySumNew = sum(min(vec,0));
    a_vec = a_vec + eta * ySumNew;              %Update rule
    thresh = eta*abs(ySumNew);
end