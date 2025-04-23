function [times] = exp_timing_function(M,node_number, timeframes)
%returns a vector of times in order of the experiment no. of the size:
%no_exp x 2 in the order [my expm, naive expm]
addpath("NBTmethods_GPU\")
addpath("NBTmethods\")
%test: argument number
% if nargin < 2
%     error("ERROR: Insufficient inputs provided");
% end
times = [];
% %test: file exists
% file_name = join([cd, sprintf("M matrices\\Nodes_%i", node_number)], '\\');
% if ~isfolder(file_name)
%     error(sprintf("ERROR: %s not found!",file_name ));
% end

% directory = dir(file_name);
% directory = directory(~[directory.isdir]);
% no_experiments = numel(directory);

% %load the experiments in order
% for i = 1:no_experiments
% 
%    matrix_file_name = join([file_name, sprintf("Nodes_%i_Timeframes_%i_Experiment_%i",node_number, timeframes,i)],'\\'); %define the matrix filename
% 
% try
%     load(matrix_file_name) %load the matrix
% catch
%     error(sprintf("ERROR: %s matrix file not found.", matrix_file_name))
% end



    alpha = 1/(size(M,1)+1); %choose an attenuation factor guaranteed to converge.
    
    
    %%%%% My exp timing goes here.
    % M = double(M>0);
    % syms t
    t = alpha;


    % disp('Creating edge-level matrices...')
    [L,S,B,W,R] = build_edge_level_matrices(M, node_number,timeframes);
    % disp('Done!')
   
    

    if isa(t, 'sym')
        L = sym(gather(L));
        R = sym(gather(R));
        S = sym(gather(S));
        B = sym(gather(B));
        W = sym(gather(W));
        M = sym(gather(M));
        Identity = sym(eye(size(B)));
        disp('WARNING: Converted matrices are now symbolic')
    else
        Identity = speye(size(B));
        Identity = gpuArray(Identity);
    end
    
%     %-------------------------------------------------Exp----------------------------------------------------------
   
 


    
    v = ones(size(W,2),1);
    W = gather(W);
    W = sparse(W);
    tic
    exp_edge = expmv(W,v,t);
    edge_time = toc;
    
    v = ones(size(M,2),1);
    
    tic
    exp_node = expmv(M,v,t);
    node_time = toc;
    
    times = [times; node_time, edge_time, node_number, nnz(W)];
    

    % [node_time,edge_time]
%     %---------------------------------------------------------------------------------------------------------------
%     % % NBTW
%     try
%         tic
%         edge_nbtw = sqrt(S)*R*ones(size(R,2),1);
%         resolvent = (Identity - t*B);
%         
%         edge_nbtw =resolvent \ edge_nbtw;
%         edge_nbtw = sqrt(S)*edge_nbtw;
%         edge_nbtw = t*L.'*edge_nbtw;
%         edge_nbtw = edge_nbtw + ones(node_number,1);
%         edge_time = toc
%     catch
%         disp('Migrating matrices to CPU...')
%         L = gather(L);
%         R = gather(R);
%         S = gather(S);
%         B = gather(B);
%         M = gather(M);
%         Identity = gather(Identity);
%         disp("Done!")
%         tic
%         edge_nbtw = sqrt(S)*R*ones(size(R,2),1);
%         resolvent = (Identity - t*B);
%         edge_nbtw =resolvent \ edge_nbtw;
%         edge_nbtw = sqrt(S)*edge_nbtw;
%         edge_nbtw = t*L.'*edge_nbtw;
%         edge_nbtw = edge_nbtw + ones(node_number,1);
%         edge_time = toc
%     end
       
    % tic
    % node_nbtw = NewnodeNBTW_gpu(M,t,node_number,timeframes); %testinv is wrong
    % node_time = toc
    % true_ans = NewnodeNBTW(gather(M),t,node_number, timeframes);
    % true_ans = true_ans*ones(size(true_ans,2),1);
    % true_ans = true_ans(1:node_number);
    % scatter(node_nbtw, edge_nbtw, 'blue')
    % 
    % xlabel("node")
    % ylabel("edge")
    % 
%     %---------------------------------------------------------------------------------------------------------------  


    

% end

    
    
    
    





end
