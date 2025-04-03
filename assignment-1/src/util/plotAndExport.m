function plotAndExport(t, x, xlabelText, ylabelText, titleText, outputDir, fileName, legendText)
    fullPath = fullfile(outputDir, fileName);

    % Check if the output directory exists, if not, create it
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    figure;
    plot(t, x);
    xlabel(xlabelText, 'Interpreter', 'latex');
    ylabel(ylabelText, 'Interpreter', 'latex');
    title(titleText);
    if nargin == 8
        legend(legendText, 'Interpreter', 'latex');
    end

    exportgraphics(gcf, fullPath, 'ContentType', 'vector');
end