function [Ls,Ss,Bs,Ws,Rs] = build_edge_level_matrices(M, number_nodes,timeframes)


As = cell(1, timeframes);

if number_nodes > 400
    useGPU = false;
else
    useGPU = false;
end

for i = 1:timeframes %can change to parfor
    As{i} = M(number_nodes*(i-1)+1: number_nodes*i,number_nodes*(i-1)+1: number_nodes*i);
end

Ls = cell(1,timeframes);
Rs = cell(1,timeframes);
Ss = cell(1,timeframes);

Bs = cell(timeframes, timeframes);
Ws = cell(timeframes, timeframes);

for i = 1:timeframes
    
    [source, target, weight] = find(As{i});
    no_edges = size(weight,1);
    L = sparse(1:no_edges, source,1, no_edges, number_nodes);
    R = sparse(1:no_edges, target,1, no_edges, number_nodes); 
    S = sparse(1:no_edges, 1:no_edges, weight, no_edges, no_edges);
   
    Ls{i} = L;
    Rs{i} = R;
    Ss{i} = S;
    clear L R S
end
if ~useGPU
    Ls = cellfun(@gather, Ls, 'UniformOutput', false);

    Rs = cellfun(@gather, Rs, 'UniformOutput', false);

    Ss = cellfun(@gather, Ss, 'UniformOutput', false);
end
for i = 1:timeframes
    for j = 1:timeframes
        [i,j];
        if j < i
            if useGPU
                Bs{i,j} = gpuArray(sparse(size(Bs{1,i},2), size(Bs{j,j},1))); %if i > j, then j,j is done as is 1,i
                Ws{i,j} = gpuArray(sparse(size(Bs{1,i},2), size(Bs{j,j},1)));
            else
                Bs{i,j} = sparse(size(Bs{1,i},2), size(Bs{j,j},1)); %if i > j, then j,j is done as is 1,i
                Ws{i,j} =sparse(size(Bs{1,i},2), size(Bs{j,j},1));
            end
        else
            
        Wij = Ss{i}*Rs{i};
        Wij = Wij*Ls{j}.';
        Wij = Wij*Ss{j};
        Ws{i,j} = sqrt(Wij);
        Wji = Ss{j}*Rs{j};
        Wji = Wji*Ls{i}.';
        Wji = Wji*Ss{i};
        
        Wij = Wij - sqrt(Wij.*Wji.');
        Bs{i,j} = sqrt(Wij);
        
        clear Wij Wji 
        end
    end
end
Bs = cellfun(@gather, Bs, 'UniformOutput', false);
Bs = cell2mat(Bs);
Ws = cellfun(@gather, Ws, 'UniformOutput', false);
Ws = cell2mat(Ws);
try
    Bs = full(gpuArray(Bs));
    Ws = full(gpuArray(Ws));
end
Rs = cellfun(@gather, Rs, 'UniformOutput', false);
Rs = vertcat(Rs{:});



Ls = cellfun(@gather, Ls, 'UniformOutput', false);
Ls = vertcat(Ls{:});


Ss = cellfun(@gather, Ss, 'UniformOutput', false);
Ss = blkdiag(Ss{:});

if useGPU
    Ls  = gpuArray(Ls);
    Ss = gpuArray(Ss);
    Rs  = gpuArray(Rs);
end



end