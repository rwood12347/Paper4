function result = circleinv_gpu(A,nodes,times)

result = zeros(size(A), 'gpuArray');


if isa(A,'sym')
    result = sym(gather(result));
end

dim = nodes*ones(times,1);
% A = mat2cell(A,dim,dim); %to delete
% result = mat2cell(result,dim,dim); %to delete
    for i = 1:times
        row = times-i + 1;
        
        %compute the diagonal
        diag_result  = ones(nodes,nodes)./A((row-1)*nodes+1: row*nodes,(row-1)*nodes+1: row*nodes);
        diag_result(isinf(diag_result)|isnan(diag_result)) = 0;
        result((row-1)*nodes+1: row*nodes,(row-1)*nodes+1: row*nodes) = diag_result;
        
        for j = row+1:times
            temp = zeros(nodes,nodes, 'gpuArray');
            if isa(A, 'sym')
                temp = sym(gather(temp));
            end
            for k = row+1:j
               %[row,k,row,k,k,j]
                % temp = temp - A{row,k}.*result{k,j}; %to delete
                temp = temp - A((row-1)*nodes+1:row*nodes, (k-1)*nodes+1:k*nodes).*result((k-1)*nodes + 1: k*nodes, (j-1)*nodes+1:j*nodes);
            end
            
            % temp= temp./A{row,row}; %to delete
            temp = temp ./ A((row-1)*nodes+1: row*nodes,(row-1)*nodes+1: row*nodes);
            temp(isinf(temp)|isnan(temp)) = 0;
            % result{row,k}  = temp; %to delete
            result((row-1)*nodes+1: row*nodes, (k-1)*nodes + 1:k*nodes) = temp;
        end


    end
% A_density = cell2mat(A);
% A_density = A_density./A_density;
% A_density(isinf(A_density)|isnan(A_density)) = 0;
% 
% try
%     result_full = cell2mat(result);
% catch
try
  result = cell2sym(result);
end
        
% end
% result = result.*A_density;

    
end
