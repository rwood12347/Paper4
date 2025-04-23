%C:\Users\rzmwo\Desktop\Spyros Experiment 2024\Multi_run....

% %Print device info:
function M = matrix_preparation(number_of_nodes, seed,isSparse, density_constant,number_of_timeframes)


rng(seed)

%number of experiments for the average = 1  0





%Create batches of dense matrices

M = zeros(number_of_nodes*number_of_timeframes,number_of_nodes*number_of_timeframes,"gpuArray");
        
        for timeframe = 1:number_of_timeframes
            
            %Print:
           % fprintf("Nth timeframe: %i \n", timeframe)            

            % generate the initial random matrix
            experiment_matrix = rand(number_of_nodes,number_of_nodes, "gpuArray");

            % generate a sparsity filter
            if ~isSparse
                density_constant_temp = density_constant;
            else
                density_constant_temp = density_constant/number_of_nodes;
            end
            filter = experiment_matrix < density_constant_temp;

            %apply the filter
            experiment_matrix = experiment_matrix.*filter;
            % experiment_matrix = round(gather(experiment_matrix),2);
            
            experiment_matrix = gpuArray(experiment_matrix);

            % experiment_matrix = sprand(number_of_nodes, number_of_nodes, density_constant);
            experiment_matrix = gpuArray(full(experiment_matrix));
            experiment_matrix = triu(experiment_matrix) + triu(experiment_matrix).';
            experiment_matrix = experiment_matrix - diag(diag(experiment_matrix));
            experiment_matrix = experiment_matrix ./ density_constant;
            clear filter
            %build M:
            to_add = repmat(experiment_matrix, timeframe,1);
            clear experiment_matrix
            M( 1: number_of_nodes*timeframe, number_of_nodes*(timeframe-1)+1: number_of_nodes*timeframe) = to_add;

        end
        
        %save the M matrix]
        % save_path = join([cd,sprintf("M matrices\\Nodes_%i\\",number_of_nodes) ], '\\');
        % save_name = sprintf("Nodes_%i_Timeframes_%i_Experiment_%i", number_of_nodes,number_of_timeframes, experiment_no);
    %     try
    %        if ~isfolder(save_path)
    %            mkdir(save_path);
    %             fprintf('Folder created: %s\n', save_path);
    %        end
    %         % M = sym(gather(M));            %For symbolic matrices
    %         save(save_path+save_name, 'M')
    %         fprintf('Successfully saved M to %s', save_path+save_name)
    %     catch
    % 
    %         fprintf('Failed to save M')
    %     end
    %     display('Clearing M...')    
    %     clear M
    % 
    %     display('M cleared!')
    % 
    % end
end
