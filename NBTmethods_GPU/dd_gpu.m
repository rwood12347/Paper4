function A = dd_gpu(A,nodes,times)

for t_one = 1:times
    for t_two = t_one:times
        % [(t_one-1)*nodes + 1, t_one*nodes,(t_two-1)*nodes + 1, t_two*nodes] % Indices

        slice = A((t_one-1)*nodes + 1: t_one*nodes,(t_two-1)*nodes + 1: t_two*nodes);
        slice = diag(diag(slice));
        A((t_one-1)*nodes + 1: t_one*nodes,(t_two-1)*nodes + 1: t_two*nodes) = slice;
        
    end
end
end
