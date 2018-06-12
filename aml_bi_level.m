function M_opt = aml_bi_level(X, y, alpha, beta)
% function M_opt = aml_bi_level(X, y, alpha, beta);
% Input: X is 2d by N, y is in {-1, 1}^N and parameters alpha, beta;
% Output: metric M is d by d.

    [D, N] = size(X);
    d = D/2;
    M = eye(d);
%     M = rand(d, d);
%     M = M'*M;
%     [M, X] = gpu_trans(M, X);
    rho_init = 1e-3;
    A_X = X(:, y == 1);
    A_hat = A_X(1:d, :) - A_X(d+1:2*d, :);
    A = A_hat * A_hat';
    B_X = X(:, y == -1);
    B_hat = B_X(1:d, :) - B_X(d+1:2*d, :);
    B = B_hat * B_hat'; 
    X1 = X(1:d, :);
    X2 = X(d+1:2*d, :);
    X_hat = X1 - X2;
    max_iter = 100;
    relative_error = 1e-10;
    gap = 1;
    M_opt = M;
    obj_opt = Inf;

    for i = 1 : max_iter
        obj_cur = obj(M);
        if obj_cur < obj_opt
            obj_opt = obj_cur;
            M_opt = M;
        end
        if mod(i, gap) == 0
            fprintf('Iteration %d:   obj_value = %f (optimal obj_value = %f)\n',...
                i, obj_cur, obj_opt);
        end
        if i >= 2 && abs(obj_cur - obj_pre)/obj_pre < relative_error
            break;
        end
        gradient_cur = obj_gradient(M);
        max_modify = 50;
        rho_cur = rho_init;
        shrinkage = 0.5;
        for t = 0: max_modify
            M_cur = M - rho_cur * gradient_cur;
            [~, lambda_test] = eig(M_cur);
            if obj(M) - obj(M_cur) < rho_cur * shrinkage * norm(gradient_cur,'fro')^2 ||...
                any(diag(lambda_test)<=0) || ~isreal(lambda_test)
                rho_cur = rho_cur*shrinkage;
            else
                M = M_cur;
                break;
            end
        end
        
        if mod(i, gap) == 0
            fprintf('Shrinkage = %d \n', t);
        end
        obj_pre = obj_cur;
    end
    fprintf('Iterations are compeleted after %d steps, and optimal obj_value = %f\n', i, obj_opt);
    M_closed = A^(-1/2)*(A^(1/2)*B*A^(1/2))^(1/2)*A^(-1/2);
    obj_closed = obj(M_closed);
    fprintf('obj value of closed-form solution = %f\n', obj_closed);
    if obj_closed  < obj_opt
        M_opt = M_closed;
        fprintf('special point is used!\n');
    end

%%
    function value = obj(M)
    % L_g(M, X, Y) = sum_i Dist_M(x_i, x_i') + sum_i Dist_M(x_i, x_i')
    % +++
    % L_g(M, F(M), y) = sum_i x_i^hat'*U*h(Lambda)U'*x_i_hat
        [U, lambda] = eig(M);
        left = X_hat'*U;
        right = left';
        lambda = diag(lambda);
		h_lambda = h_y(lambda);
        left = left .* h_lambda';
        value_2 = beta^2 * sum(sum(left'.*right));
        value = sum(sum(A_hat.*(M*A_hat))) + ...
             sum(sum(B_hat.*(M\B_hat))) + alpha * value_2;
        value = value/N;
    end
	
	function grad = obj_gradient(M)
		[U, lambda] = eig(M);
		lambda = diag(lambda);
		grad_2 = 0;
        for j = 1 : length(lambda)
			right = U(:, j) * U(:, j)';
			T = X_hat' * U(:, j);
			left = sum(h_y_derivate(lambda(j))' .* sum(T .* T, 2));
			grad_2 = grad_2 + beta^2 * left * right;
            
            lambda_t = lambda - lambda(j);
            pos_indexs = lambda_t ~= 0;
			lambda_pos  = lambda_t(pos_indexs);
			lambda_t(pos_indexs) = 1./lambda_pos;
			left = 2 * (U * diag(lambda_t) * U') ...
                * (X_hat .* repmat(h_y(lambda(j)), d, 1) * X_hat');
			grad_2 = grad_2 + beta^2 * left * right;
        end
        grad = A - (M\B)/M + alpha * grad_2;
        grad = grad/N;
    end
	
	function vectors = h_y(lambdas)
	% Input: lambdas is d dimension
	% Output: vectors is d by N.
		d_lambda = length(lambdas);
		Y = repmat(y', d_lambda, 1);
		lambdas = repmat(lambdas, 1, N);
        vectors = (lambdas.^(3+2*Y)) ./ (2 + beta^2 * lambdas.^(1+Y)).^2;
	end
	
	function vectors = h_y_derivate(lambdas)
	% Input: lambdas is d dimension
	% Output: vectors is d by N.
		d_lambda = length(lambdas);
		Y = repmat(y', d_lambda, 1);
		lambdas = repmat(lambdas, 1, N);
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