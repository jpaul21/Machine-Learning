function [np, nq] = matrix_factorization(R,P,Q,K, steps, alpha, beta)

R_est = P * Q;
error = abs(R - R_est);

[ii, jj] = find(~R);
for i = 1:length(ii)
    error(ii(i), jj(i)) = 0;
end


for i = 1:steps
    P = P + alpha .* (2 .* error * transpose(Q) - beta .* P);
    Q = Q + alpha .* (2 .* transpose(transpose(error) * P) - beta .* Q);
    error = (R - P*Q);
end

np = P;
nq = Q;