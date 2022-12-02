%% Expectation Maximization algorithm
%% Parameter initialization
load('hw3.mat');
D = hw3_1;
label = [ones(1,50), 2*ones(1,50)];
label_1 = label(1,1:50);
label_2 = label(1,51:end);
%weight = [0.5; 0.5];
% Check on covariance
theta1 = [mean(D(:,1:50), 2), cov(D(1,1:50), D(2,1:50))];
theta2 = [mean(D(:,51:100), 2), cov(D(1,51:100), D(2,51:100))];
%% EM algorithm
% fill vector based on likelihood
for j = 1:5
    D1 = [;];
    D2 = [;];
    for i = 1:length(label)
        likelihood1 = likefun1(D(:, i), theta1);
        likelihood2 = likefun2(D(:, i), theta2);
        if likelihood1 > likelihood2
            label(i) = 1;
            D1(:, end+1) = D(:, i);
        else 
            label(i) = 2;
            D2(:, end+1) = D(:, i);
        end
    end
    D = [D1, D2];
    padding = max(length(D1), length(D2));
    if length(D1) > length(D2)
        pad = zeros(2,padding-length(D2));
        D2 = [D2, pad];
    elseif length(D2) > length(D1)
        pad = zeros(2,padding-length(D1));
        D1 = [D1, pad];
    else
    end
    C_new = cov(D1(1,:), D1(2,:));
    C_new1 = cov(D2(1,:), D2(2,:));
%    ro0 = length(D1) / (length(D1) + length(D2));
%    ro1 = (1 - ro0);
    theta1 = [mean(D1, 2), C_new];
    theta2 = [mean(D2, 2), C_new1];
end
%% likelihood functions
% likelihood (p(x|theta))
%likelihood2 = -0.5 * log(ro_fun1*2*pi*var_fun1) - (1/(2*var_fun1)) * transpose(x-mu_fun1)*(x-mu_fun1);
%likelihood1 = 1 / (sqrt(2*pi*det(var_fun))) *exp(- 0.5 * transpose(x-mu_fun)*inv(var_fun)*(x-mu_fun));
%-(length(x)/2)*log(2*pi) - length(x)*log(sqrt(var_fun)) -0.5*(transpose(x-mu_fun)*(x-mu_fun)/var_fun);
function likelihood1 = likefun1(x, theta)
    mu_fun = theta(:,1);
    var_fun = theta(:,2:3);
%    ro_fun = theta(1,4);
    likelihood1 = 1 / (2*pi*(sqrt(det(var_fun)))) *exp(- 0.5 * transpose(x-mu_fun)*inv(var_fun)*(x-mu_fun));
end
function likelihood2 = likefun2(x, theta)
    mu_fun1 = theta(:,1);
    var_fun1 = theta(:,2:3);
%    ro_fun1 = theta(2,4);
    likelihood2 = 1 / (2*pi*(sqrt(det(var_fun1)))) *exp(- 0.5 * transpose(x-mu_fun1)*inv(var_fun1)*(x-mu_fun1));
end