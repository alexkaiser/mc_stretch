% Plot script for histograms. 

% Read these values from a file 
% name = 'Stretch move'; 
% totalSamples = 1000000; 
% K = 100;
% M = totalSamples / K; 
% N = 10; 
% pdf_num

% Run the parameter load script
load_parameters

totalSamples = M * K; 
numSaved = length(indicesToSave); 

% look at each of the saved variables 
for j = 1:numSaved
    
    idx = indicesToSave(j) ; 

    % get the precomputed historgram data 
    eval(sprintf('histogram_data_%d', idx));
    figure ; 
    
    % Swap these lines to add error bars 
    % errorbar(centers, fhat, sigma_f_hat, 'r') ;
    bar(centers, fhat, 1.0) ;
    hold on
           
    title(sprintf('%s run:\nHistogram of X_{%d} of %d\nSteps, M = %d, Number of walkers = %d, Total Samples = %d', ...
        name, idx, N, M, K, totalSamples)); 

    legend('Histogram of samples'); 
     
    
    hold off 
end 

% scatter plot for 2d dimensions 
if ((N == 2) && (numSaved == 2))
    figure; 
    plot(samples(1,:), samples(2,:), '.k', 'MarkerSize', 0.001) ; 
end
