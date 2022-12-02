D = [1 -3 4 -1 0 5 -1 3; 2 -1 5 1 -2 2 -4 1];
m = mean(D, 2);
diff = D - m;
[rows, columns] = size(D);
S = 0;
for i = 1:columns
    S = S + diff([1 2], i) * transpose(diff([1 2], i));
end

[e, f] = eig(S);
[~, I] = max(f, [], 2);
[~, I2] = max(I);
A = e(:, I2);
ak = transpose(A) * diff;
P1 = m + ak .* A;

m1 = mean(D(:,1:4), 2);
m2 = mean(D(:, 5:8), 2);

diff1 = D(:, 1:4) - m1;
diff2 = D(:, 5:8) - m2;

S1 = 0;
S2 = 0;
for j = 1:4
    S1 = S1 + diff1([1 2], j) * transpose(diff1([1 2], j));
    S2 = S2 + diff2([1 2], j) * transpose(diff2([1 2], j));
end

Sw = S1 + S2;
w = inv(Sw) *(m1 - m2);
yk = transpose(w) * D;
P2 = yk .* w;
