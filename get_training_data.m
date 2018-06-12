function [X, y] = get_training_data(Examples, Labels, N)
% Input: Examples (d x n), Labels (n), N is the number of training pair
% Output: X (2d x N), y (N)
    [d, n] = size(Examples);
    indexs = randperm(n^2, N);
    X = zeros(2*d, N);
    y = zeros(N, 1);
    for i=1: length(indexs)
        u = indexs(i)/n;
        v = mod(indexs(i), n);
        p = floor(u) + (v~=0);
        q = v + 1;
        X(:, i) = [Examples(:, p);Examples(:, q)];
        y(i) = (Labels(p) == Labels(q));
    end
    y(y == 0) = -1;
end