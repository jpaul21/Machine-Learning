clear

d11=[2;2]*ones(1,70)+2.*randn(2,70);
d12=[-2;-2]*ones(1,30)+randn(2,30);
d1=[d11,d12];

d21=[3;-3]*ones(1,50)+randn([2,50]);
d22=[-3;3]*ones(1,50)+randn([2,50]);
d2=[d21,d22];

hw5_1=d1;
hw5_2=d2;

save hw5.mat hw5_1 hw5_2

x1=hw5_1;
x2=hw5_2;

plot(x1(1,:),x1(2,:),'o',x2(1,:),x2(2,:),'*')
%% Stochastic Backpropagation 
% Hidden nodes (hn) = 10, Input Nodes (I) = 2, Input pattern dimension = 2
% Convergence criterion = 0.1
% Convergence rate (eta) = 0.1
% Activation function (f(net)) = a*tanh(b*net), a = 1.716, b = 2/3
% w1 target vector = [1;-1], w2 target vector  = [-1,1]
% Use standardized input patterns and random uniform weights initialization

% 1. samples
X = [x1, x2];

% 2. target matrix
w1_targ = [1;-1];   % class 1 target vector
w2_targ = [-1;1];   % class 2 target vector
targ_matrix_1 = repmat(w1_targ,1,100);
targ_matrix_2 = repmat(w2_targ, 1, 100); % target matrix
targ_matrix = [targ_matrix_1, targ_matrix_2];

% 3. Normalize
%X_norm = [normalize(X(1,:)); normalize(X(2,:))];
X_norm = zscore(X,0,2);

% 4. Initialize parameters
hn = 10;            % hidden nodes
Ni = 2;             % Input nodes
No = 2;             % Output nodes
eta = 0.1;          % convergence rate
theta = 0.1;        % Convergence criterion
J = 1;              % Initialize error rate 

f = @(net) 1.716 * tanh(0.6667 * net);   % activation function
f_1 = @(net) 1.716 * (0.6667 * sech(0.6667 * net) .^2); % activation derivative

% 5. Initialize weights
toHidden_lower = -1 / sqrt(Ni);          % > w(j,i)
toHidden_upper = 1 / sqrt(Ni);           % < w(j,i)

fromHidden_lower = -1 / sqrt(hn);        % > w(k,j)
fromHidden_upper = 1 / sqrt(hn);         % < w(k,j)

toH_weights = toHidden_lower + (toHidden_upper-toHidden_lower)*rand(10,2);
fromH_weights = fromHidden_lower+(fromHidden_upper-fromHidden_lower)*rand(2,10);
count = 0;
while(J > theta)
    count = count + 1;
    % 6. Random sample
    randX = randi(200);

    % 7. Net(j)
    % = transpose(X_norm(:,randX))*transpose(toH_weights)+1
    net_j = ((toH_weights) * X_norm(:, randX)) + 1;
    
    % 8. Hidden node output
    Y = f(net_j);
    
    % 9. Net(k)
    net_k = transpose(Y) * transpose(fromH_weights) + 1;
    
    % 10. Output
    Z = f(net_k);
    Z = transpose(Z);
    
    % 11. Sensitivity delK
    del_k = targ_matrix(:,randX) - Z;
    delta_k = transpose(del_k) * transpose(f_1(net_k));
    
    % 12. Sensitivity delJ
    delta_j = (delta_k * fromH_weights) * f_1(net_j);
    
    % 13. Update (from/to)H_weights
    fromH_weights = fromH_weights + transpose(eta * delta_k * Y);
    toH_weights = toH_weights + transpose(eta * transpose(delta_j) * X_norm(:, randX));
    
    % 14. Training Error
    J = 0.5 * norm(targ_matrix(:,randX) - Z, 2).^2;
end

%% Classification
D_test = [2, -3, -2, 3; 2, -3, 5, -4];
results = [];
%test_vec = 10;          % Index of test vector in X.
for b = 1:4
    test_netJ = transpose(D_test(:,b)) * transpose(toH_weights) + 1;
    test_y = f(test_netJ);
    test_netK = test_y * transpose(fromH_weights) + 1;
    test_Z = f(test_netK);
    results(b,:) = test_Z;
end
transpose(results)
