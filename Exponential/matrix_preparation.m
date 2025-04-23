%C:\Users\rzmwo\Desktop\Spyros Experiment 2024\Multi_run....

% %Print device info:
function M = matrix_preparation(number_of_nodes, seed,isSparse, density_constant,number_of_timeframes)


rng(seed)

%number of experiments for the average = 1  0





%Create batches of dense matrices

M = zeros(number_of_nodes*number_of_timeframes,number_of_nodes*number_of_timeframes);
        
        for timeframe = 1:number_of_timeframes
            
            %Print:
           % fprintf("Nth timeframe: %i \n", timeframe)            

            % generate the initial random matrix
            experiment_matrix = rand(number_of_nodes,number_of_nodes);

            % generate a sparsity filter
            if ~isSparse
                density_constant_temp = density_constant;
            else
                density_constant_temp = density_constant/number_of_nodes;
            end
            filter = experiment_matrix < density_constant_temp;

   
            experiment_matrix = experiment_matrix.*filter;

            experiment_matrix = triu(experiment_matrix) + triu(experiment_matrix).';
            experiment_matrix = experiment_matrix - diag(diag(experiment_matrix));
            experiment_matrix = experiment_matrix ./ density_constant;
            clear filter
            %build M:
            to_add = repmat(experiment_matrix, timeframe,1);
            clear experiment_matrix
            M( 1: number_of_nodes*timeframe, number_of_nodes*(timeframe-1)+1: number_of_nodes*timeframe) = to_add;

        end
        

end
