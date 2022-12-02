input_file = readtable('EE627A_HW1_Data.csv', VariableNamingRule='preserve');
data = table2array(input_file(:, [2,3,4,5]));
[coeff, score, lambda] = pca(data);
lambda_total = sum(lambda);
proportions = lambda / lambda_total;
cummul = [proportions(1)];
for i = 2:4
    cummul = [cummul; cummul(i-1) + proportions(i)];
end

proportions
cummul

pc1 = data * coeff(:,1);
pc2 = data * coeff(:,2);
plot(pc1, '.')
title('PC1')
figure()
plot(pc2, '.')
title('PC2')