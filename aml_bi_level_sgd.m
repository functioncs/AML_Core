function M_opt = aml_bi_level_sgd(X, y, alpha, beta)
% function M_opt = aml_bi_level(X, y, alpha, beta);
% Input: X is 2d by N, y is in {-1, 1}^N and parameters alpha, beta;
% Output: metric M is d by d.

    [D, N] = size(X);
    d = D/2;
    M = eye(d);
%     M = rand(d, d);
%     M = M'*M;
%     [M, X] = gpu_trans(M, X);
    rho_init = 1;
    A_X = X(:, y == 1);
    A_hat = A_X(1:d, :) - A_X(d+1:2*d, :);
    A = A_hat * A_hat';
    B_X = X(:, y == -1);
    B_hat = B_X(1:d, :) - B_X(d+1:2*d, :);
    B = B_hat * B_hat'; 
    X1 = X(1:d, :);
    X2 = X(d+1:2*d, :);
    X_hat = X1 - X2;
    batch_size = 50;%%%%%%%%%%%%%%%%%%%%%
    max_epochs = 200;
    gap = 1;
    M_opt = M;
    obj_opt = Inf;
    for i = 1 : max_epochs
        obj_cur = obj(M);
        if obj_cur < obj_opt
            obj_opt = obj_cur;
            M_opt = M;
        end
        if mod(i, gap) == 0
            fprintf('Iteration %d:   obj_value = %f (optimal obj_value = %f)\n',...
                i, obj_cur, obj_opt);
        end
        shuff = randperm(N);
        X = X(:, shuff);
        y = y(shuff);
        shrinkage_record = 0;
        for t = 1 : N/batch_size
            batch = (t-1)*batch_size + 1: t*batch_size;
            X_batch = X(:, batch);
            y_batch = y(batch);
            gradient_cur = obj_batch_gradient(M, X_batch, y_batch);
            max_modify = 20;
            rho_cur = rho_init;
            shrinkage = 0.5;
            for l = 1: max_modify
                M_cur = M - rho_cur * gradient_cur;
                [~, lambda_test] = eig(M_cur);
                if obj_batch(M, X_batch, y_batch) - obj_batch(M_cur, X_batch, y_batch) <...
                        rho_cur*shrinkage*norm(gradient_cur,'fro')^2 || any(diag(lambda_test)<=0)
                    rho_cur = rho_cur*shrinkage;
                else
                    M = M_cur;
                    break;
                end
            end
            if l > shrinkage_record
                shrinkage_record = l;
            end
        end
        if mod(i, gap) == 0
            fprintf('Max shrinkage count: %d: \n',shrinkage_record);
        end
    end
    fprintf('Iterations are compeleted after %d, steps and optimal obj_value = %f\n', i, obj_opt);
%%
    function value = obj(M)
    % L_g(M, X, Y) = sum_i Dist_M(x_i, x_i') + sum_i Dist_M(x_i, x_i')
    % +++
    % L_g(M, F(M), y) = sum_i x_i^hat'*U*h(Lambda)U'*x_i_hat
        [U, lambda] = eig(M);
        left = X_hat'*U;
        right = left';
        lambda = diag(lambda);
		h_lambda = h_y(lambda, y);
        left = left .* h_lambda';
        value_2 = beta^2 * sum(sum(left'.*right));
        value = sum(sum(A_hat.*(M*A_hat))) + ...
             sum(sum(B_hat.*(M\B_hat))) + alpha * value_2;
        value = value/N;
    end

    function value = obj_batch(M, X_batch, y_batch)
        A_batch_X = X_batch(:, y_batch == 1);
        A_batch_hat = A_batch_X(1:d, :) - A_batch_X(d+1:2*d, :);
        B_batch_X = X_batch(:, y_batch == -1);
        B_batch_hat = B_batch_X(1:d, :) - B_batch_X(d+1:2*d, :);
        X1_batch = X_batch(1:d, :);
        X2_batch = X_batch(d+1:2*d, :);
        X_batch_hat = X1_batch - X2_batch;
        
        [U, lambda] = eig(M);
        left = X_batch_hat'*U;
        right = left';
        lambda = diag(lambda);
		h_lambda = h_y(lambda, y_batch);
        left = left .* h_lambda';
        value_2 = beta^2 * sum(sum(left'.*right));
        value = sum(sum(A_batch_hat.*(M*A_batch_hat))) + ...
             sum(sum(B_batch_hat.*(M\B_batch_hat))) + alpha * value_2;
        value = value/N;
    end

    function [A_batch, B_batch, X_batch_hat] = batch_data(X_batch, y_batch)
        A_batch_X = X_batch(:, y_batch == 1);
        A_batch_hat = A_batch_X(1:d, :) - A_batch_X(d+1:2*d, :);
        A_batch = A_batch_hat * A_batch_hat';
        B_batch_X = X_batch(:, y_batch == -1);
        B_batch_hat = B_batch_X(1:d, :) - B_batch_X(d+1:2*d, :);
        B_batch = B_batch_hat * B_batch_hat';
        X1_batch = X_batch(1:d, :);
        X2_batch = X_batch(d+1:2*d, :);
        X_batch_hat = X1_batch - X2_batch;
    end

	function grad = obj_batch_gradient(M, X_batch, y_batch)
		[U, lambda] = eig(M);
		lambda = diag(lambda);
        [A_batch, B_batch, X_batch_hat] = batch_data(X_batch, y_batch);
        
		grad_2 = 0;
        for j = 1 : length(lambda)
			right = U(:, j) * U(:, j)';
			T = X_batch_hat' * U(:, j);
			left = sum(h_y_derivate(lambda(j), y_batch)' .* sum(T .* T, 2));
			grad_2 = grad_2 + beta^2 * left * right;
            
            lambda_t = lambda - lambda(j);
            pos_indexs = lambda_t ~= 0;
			lambda_pos  = lambda_t(pos_indexs);
			lambda_t(pos_indexs) = 1./lambda_pos;
			left = 2 * (U * diag(lambda_t) * U') ...
                * (X_batch_hat .* repmat(h_y(lambda(j), y_batch), d, 1) * X_batch_hat');
			grad_2 = grad_2 + beta^2 * left * right;
        end
        grad = A_batch - (M\B_batch)/M + alpha * grad_2;
        grad = grad/N;
    end
	
	function vectors = h_y(lambdas, y_batch)
	% Input: lambdas is d dimension, y_batch is d_y_batch
	% Output: vectors is d by batch_size.
		d_lambda = length(lambdas);
        d_y_batch = length(y_batch);
		Y = repmat(y_batch', d_lambda, 1);
		lambdas = repmat(lambdas, 1, d_y_batch);
        vectors = (lambdas.^(3+2*Y)) ./ (2 + beta^2 * lambdas.^(1+Y)).^2;
	end
	
	function vectors = h_y_derivate(lambdas, y_batch)
	% Input: lambdas is d dimension, y_batch is d_y_batch
	% Output: vectors is d by batch_size.
		d_lambda = length(lambdas);
        d_y_batch = length(y_batch);
		Y = repmat(y_batch', d_lambda, 1);
		lambdas = repmat(lambdas, 1, d_y_batch);
		vectors1 = ((3+2*Y).*lambdas.^(2+2*Y).*(2 + beta^2 * lambdas.^(1+Y)).^2 - ...
		lambdas.^(3+2*Y) .* (2*(2 + beta^2 * lambdas.^(1+Y)).* (beta^2*(1+Y).*lambdas.^Y)));
        vectors2 = (2 + beta^2 * lambdas.^(1+Y)).^4;
        vectors = vectors1./vectors2;
        if isNorI(vectors)
            fprintf('error!\n');
        end
	end

    function [M, X] = gpu_trans(M, X)
        M = gpuArray(M);
        X = gpuArray(X);
    end
    function [U, V] = get_real(U, V)
        U = real(U);
        V = real(V);
    end
    function log = isNorI(W)
        log = any(any(isnan(W))) || any(any(isinf(W)));
    end
end