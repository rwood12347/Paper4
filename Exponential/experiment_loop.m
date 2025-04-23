%Experiment loop

function all_times =  experiment_loop(experiments, min_nodes, max_nodes,node_skip, time_frames, seed, isSparse) 
save_name = join([cd, "results" ], '\\')


all_times = [];
node_sizes = linspace(min_nodes, max_nodes, node_skip);
node_sizes = floor(node_sizes)
for i = 1:size(node_sizes,2)
batch_times = [];
     for experiment = 1:experiments
         M = matrix_preparation(node_sizes(i), seed, isSparse, 0.3, time_frames);
        % fprintf('Loading %s', to_load_exp)
        times = exp_timing_function(M, node_sizes(i), time_frames);
        batch_times = [batch_times; times]; %times = [node_time, edge_time, nodes, edges]
     end
    all_times = [all_times; mean(batch_times, 1)];

end

results = table(all_times(:,1), all_times(:,2), all_times(:,3), all_times(:,4), 'VariableNames', {'NodeTime', 'EdgeTime', 'No_nodes', 'No_edges'});
save(save_name, "results")
% bar(results.EdgeTime-results.NodeTime)

end
