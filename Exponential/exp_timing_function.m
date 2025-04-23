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



    alpha = 1/(size(M,1)+1); %choose an attenuation factor guaranteed to converge.
    

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
    end
    
%     %-------------------------------------------------Exp----------------------------------------------------------
   
 


    
    v = ones(size(W,2),1);
  
    W = sparse(W);
    tic
    exp_edge = expmv(W,v,t);
    edge_time = toc;
    
    v = ones(size(M,2),1);
    
    tic
    exp_node = expmv(M,v,t);
    node_time = toc;
    
    times = [times; node_time, edge_time, node_number, nnz(W)];
    

    

    
    





end
