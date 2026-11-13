close all
clear
clc

csv = "C:\Github\CSCI-198-Nematode-Classification\results\n2_herma\record_1003271_time.csv";

% --- Put base directory as folder "openworm"
baseDir = "C:\Github\CSCI-198-Nematode-Classification\tierpsy";

% --- Put end directory as folder "results"
outputDir = "C:\Github\CSCI-198-Nematode-Classification\tierpsy\results";

% --- Find strain folders automatically
d = dir(baseDir);
isFolder = [d.isdir];
strainNames = string({d(isFolder).name});
strainNames = strainNames(~ismember(strainNames, [".", ".."]));

nskip = 0;
nsummaries = 1;   
dataset_all_summary = cell(1, nsummaries);

for j = 1:nsummaries
    dataset_all_summary{j} = table();
end

dataset_all_time = table();
dataset_strain = table();
dataset_frames = table();
dataset_fps = table();

for s = 1:numel(strainNames)
    strain = strainNames(s);
    strainPath = fullfile(baseDir, strain);
    
    csvFiles = dir(fullfile(strainPath, "*.csv"));
    
    % Numerical order
    names = {csvFiles.name};
    nums = cellfun(@(x) sscanf(x, 'record_%d.csv'), names);
    [~, idx] = sort(nums);
    csvFiles = csvFiles(idx);
    
    for k = 1:numel(csvFiles)
        % Set max frames
        max_frames = inf;

        csvPath = fullfile(strainPath, csvFiles(k).name);
        
        fprintf("Processing: %s\n", csvPath);
        
        table_original = readtable(csvPath);
        if (max_frames ~= inf)
            table_original = table_original(1:max_frames, :); 
        end
        
        if (max_frames == inf)
            max_frames = height(table_original);
        end
        
        % filter table for skipped frames
        rows_all_na = ismissing(table_original.SkeletonLength);
        table_all_na = table_original(rows_all_na, :);
        num_rows_all_na = height(table_all_na);
        % table_not_all_na = table_original(~rows_all_na, :);
        
        %{
        % interpolate points
        table_interpolate = table_original;
        
        table_interpolate.CentroidX = fillmissing(table_original.CentroidX,'linear');
        table_interpolate.CentroidY = fillmissing(table_original.CentroidY,'linear');
        
        table_interpolate.EndpointX1 = fillmissing(table_original.EndpointX1,'linear');
        table_interpolate.EndpointY1 = fillmissing(table_original.EndpointY1,'linear');
        table_interpolate.EndpointX2 = fillmissing(table_original.EndpointX2,'linear');
        table_interpolate.EndpointY2 = fillmissing(table_original.EndpointY2,'linear');
        %}
        
        % interpolate all features
        table_filled = fillmissing(table_original,'linear');
        
        % Recalculate displacement
        % Consider frame rate 
        fps = table_filled.FPS(1);
        
        dx_c = diff(table_filled.CentroidX);
        dy_c = diff(table_filled.CentroidY);
        table_filled.CentroidDisplacement = [0; hypot(dx_c, dy_c) * fps];
        
        dx_e1 = diff(table_filled.EndpointX1);
        dy_e1 = diff(table_filled.EndpointY1);
        table_filled.EndpointDisplacement1 = [0; hypot(dx_e1, dy_e1) * fps];
        
        
        dx_e2 = diff(table_filled.EndpointX2);
        dy_e2 = diff(table_filled.EndpointY2);
        table_filled.EndpointDisplacement2 = [0; hypot(dx_e2, dy_e2) * fps];
        
        % Frame skipping
        max_frameskip = 5; % was 5

        for i = 0:max_frameskip
            
            lag = i + 1;
            
            dx_c = table_filled.CentroidX(1+lag:end) - table_filled.CentroidX(1:end-lag);
            dy_c = table_filled.CentroidY(1+lag:end) - table_filled.CentroidY(1:end-lag);
            
            centroid_displacement = hypot(dx_c, dy_c) * fps / lag;  
            centroid_displacement = [zeros(lag,1); centroid_displacement];
            
            table_filled.("CentroidDisplacementFS" + i) = centroid_displacement;

            dx_e1 = table_filled.EndpointX1(1+lag:end) - table_filled.EndpointX1(1:end-lag);
            dy_e1 = table_filled.EndpointY1(1+lag:end) - table_filled.EndpointY1(1:end-lag);
            
            endpoint1_displacement = hypot(dx_e1, dy_e1) * fps / lag;  
            endpoint1_displacement = [zeros(lag,1); endpoint1_displacement];
            
            table_filled.("Endpoint1DisplacementFS" + i) = endpoint1_displacement;

            dx_e2 = table_filled.EndpointX2(1+lag:end) - table_filled.EndpointX2(1:end-lag);
            dy_e2 = table_filled.EndpointY2(1+lag:end) - table_filled.EndpointY2(1:end-lag);
            
            endpoint2_displacement = hypot(dx_e2, dy_e2) * fps / lag;  
            endpoint2_displacement = [zeros(lag,1); endpoint2_displacement];
            
            table_filled.("Endpoint2DisplacementFS" + i) = endpoint2_displacement;
        end
        
        % Recalculate endpoint separation
        table_filled.EndpointSeparation = hypot( ...
            table_filled.EndpointX1 - table_filled.EndpointX2, ...
            table_filled.EndpointY1 - table_filled.EndpointY2 );
        
        % Add total displacement
        x0_c = table_filled.CentroidX(1);
        y0_c = table_filled.CentroidY(1);
        
        x0_e1 = table_filled.EndpointX1(1);
        y0_e1 = table_filled.EndpointY1(1);
        
        x0_e2 = table_filled.EndpointX2(1);
        y0_e2 = table_filled.EndpointY2(1);
        
        table_filled.CentroidTotalDisplacement = hypot( ...
            table_filled.CentroidX - x0_c, ...
            table_filled.CentroidY - y0_c ) * fps;
        
        table_filled.Endpoint1TotalDisplacement = hypot( ...
            table_filled.EndpointX1 - x0_e1, ...
            table_filled.EndpointY1 - y0_e1 ) * fps;
        
        table_filled.Endpoint2TotalDisplacement = hypot( ...
            table_filled.EndpointX2 - x0_e2, ...
            table_filled.EndpointY2 - y0_e2 ) * fps;
        
        %
        % Center points from start
        table_filled.CentroidX = table_filled.CentroidX - x0_c;
        table_filled.CentroidY = table_filled.CentroidY - y0_c;
        
        table_filled.EndpointX1 = table_filled.EndpointX1 - x0_c;
        table_filled.EndpointY1 = table_filled.EndpointY1 - y0_c;
        
        table_filled.EndpointX2 = table_filled.EndpointX2 - x0_c;
        table_filled.EndpointY2 = table_filled.EndpointY2 - y0_c;
        %}
        
        %{
        % Reorient points from angle
        dx_es = table_filled.EndpointX2 - table_filled.EndpointX1;
        dy_es = table_filled.EndpointY2 - table_filled.EndpointY1;
        
        theta_end = mean(atan2(dy_es, dx_es));
        theta_ang = deg2rad( ...
            atan2d( ...
                mean(sind(table_filled.OrientationAngle(1:max_frames))), ...
                mean(cosd(table_filled.OrientationAngle(1:max_frames))) ...
            ) ...
        );
        theta = deg2rad(mean(table_filled.OrientationAngle));
        
        c = cos(-0);
        sn = sin(-0);
        
        x_c = table_filled.CentroidX;
        y_c = table_filled.CentroidY;
        table_filled.CentroidX = x_c.*c - y_c.*sn;
        table_filled.CentroidY = x_c.*sn + y_c.*c;
        
        x_e1 = table_filled.EndpointX1;
        y_e1 = table_filled.EndpointY1;
        table_filled.EndpointX1 = x_e1.*c - y_e1.*sn;
        table_filled.EndpointY1 = x_e1.*sn + y_e1.*c;
        
        x_e2 = table_filled.EndpointX2;
        y_e2 = table_filled.EndpointY2;
        table_filled.EndpointX2 = x_e2.*c - y_e2.*sn;
        table_filled.EndpointY2 = x_e2.*sn + y_e2.*c;
        %}
        
        %{
        % Figure
        figure('Color','w');
        hold on;
        
        % find original NaN points
        nan_centroid = isnan(table_original.CentroidX) | isnan(table_original.CentroidY);
        nan_ep1 = isnan(table_original.EndpointX1) | isnan(table_original.EndpointY1);
        nan_ep2 = isnan(table_original.EndpointX2) | isnan(table_original.EndpointY2);
        
        % Plot full interpolated trajectories
        plot(table_filled.CentroidX, table_filled.CentroidY, ...
            'g-', 'LineWidth', 1);
        
        plot(table_filled.EndpointX1, table_filled.EndpointY1, ...
            'r-', 'LineWidth', 1);
        
        plot(table_filled.EndpointX2, table_filled.EndpointY2, ...
            'b-', 'LineWidth', 1);
        
        
        % Overlay interpolated-only points in different color
        scatter(table_filled.CentroidX(nan_centroid), ...
                table_filled.CentroidY(nan_centroid), ...
                10, 'y', 'filled');  
        
        scatter(table_filled.EndpointX1(nan_ep1), ...
                table_filled.EndpointY1(nan_ep1), ...
                10, 'm', 'filled');   
        
        scatter(table_filled.EndpointX2(nan_ep2), ...
                table_filled.EndpointY2(nan_ep2), ...
                10, 'c', 'filled');   
        
        scatter(table_filled.CentroidX(1), ...
                table_filled.CentroidY(1), ...
                100, [0.7 0.5 0.3], 'pentagram', 'filled');
        
        scatter(table_filled.CentroidX(end), ...
                table_filled.CentroidY(end), ...
                100, [0.7 0.5 0.3], 'pentagram', 'filled');
        
        %{
        for i = 1:1000:height(table_filled)
            
            x_points = [ ...
                table_filled.EndpointX1(i), ...
                table_filled.CentroidX(i), ...
                table_filled.EndpointX2(i) ...
            ];
            
            y_points = [ ...
                table_filled.EndpointY1(i), ...
                table_filled.CentroidY(i), ...
                table_filled.EndpointY2(i) ...
            ];
            
            plot(x_points, y_points, 'w-', 'LineWidth', 10);
        end
        %}
        
        axis equal;
        xlabel('Worm X (um)');
        ylabel('Worm Y (um)');
        title('Trajectory with Interpolated Points Highlighted');
        legend({'Centroid','Endpoint 1','Endpoint 2', ...
                'Interp Centroid','Interp EP1','Interp EP2'}, ...
                'Location','best');
        hold off;
        %}
        
        features_local = {'SkeletonLength', ...
                      'EndpointSeparation', ...
                      'CentroidDisplacementFS0', ...
                      'Endpoint1DisplacementFS0', ...
                      'Endpoint2DisplacementFS0', ...
                      'CentroidTotalDisplacement', ...
                      'Endpoint1TotalDisplacement', ...
                      'Endpoint2TotalDisplacement', ...
                      'Tortuosity', ...
                      'NumberOfBends', ...
                      'BendLengthStd', ...
                      'WidthMean', ...
                      'AspectRatio', ...
                      'Area', ...
                      'Perimeter', ...
                      'OrientationAngle'
                      };
        
        table_features_local = table_filled(:, features_local);

        vars_to_check = setdiff(table_features_local.Properties.VariableNames, ...
                        {'OrientationAngle'});

        for k = 1:numel(vars_to_check)
            var = vars_to_check{k};
        
            x = table_features_local.(var);
        
            nneg = sum(x < 0 & ~isnan(x));
        
            if nneg > 0
                fprintf('%s: replacing %d negative values with 0\n', ...
                        var, nneg);
        
                x(x < 0) = 0;
                table_features_local.(var) = x;
            end
        end
        
        %{
        filter_two = {'CentroidX', ...
                      'CentroidY', ...
                      'EndpointX1', ...
                      'EndpointY1', ...
                      'EndpointX2', ...
                      'EndpointY2'
                      };
        
        table_features_global = table_filled(:, filter_two);
        %}

        array_features_local = table2array(table_features_local);
        
        % Create time series table

        [nframes, nfeatures] = size(array_features_local);

        % if fps is closer to 26 then factor is 5, if fps is closer to 30
        % then factor is 6
        if abs(fps - 26) < abs(fps - 30)
            fps_factor = 5.0;
        else
            fps_factor = 6.0;
        end

        nframes_fps = nframes / fps_factor;

        if isempty(dataset_all_time)
            min_frames = nframes;
            dataset_all_time = array_features_local;   % first slice
            dataset_all_time = reshape(dataset_all_time, min_frames, nfeatures, 1);
        else
            % Update min_frames if this file is shorter
            if nframes_fps < min_frames
                min_frames = floor(nframes_fps);
        
                % Trim existing data to new min_frames
                dataset_all_time = dataset_all_time(1:min_frames,:,:);
            end
        
            % Trim current recording to min_frames
            array_features_local = array_features_local(1:min_frames,:);
        
            % Add as new slice
            current_record = size(dataset_all_time,3) + 1;
            dataset_all_time(:,:,current_record) = array_features_local;
        end
        
        % Create summary statistics table

        for j = 1:nskip:nsummaries
            array_sub = array_features_local(1:j:end, :);

            features_mean = mean(array_sub, 1, 'omitnan');
            features_std  = std(array_sub, 0, 1, 'omitnan');
            features_skewness  = skewness(array_sub, 1, 1);
            features_kurtosis  = kurtosis(array_sub, 1, 1); 
            
            % Flatten all 4 statistics into one row
            array_features_local_summary = reshape( ...
                [features_mean;
                 features_std;
                 features_skewness;
                 features_kurtosis], ...
                 1, []);
            
            % Create feature names
            n = numel(features_local);
            features_local_flat = strings(1, 4*numel(features_local));
            
            for i = 1:n
                features_local_flat(4*i-3) = features_local{i} + "Mean";
                features_local_flat(4*i-2) = features_local{i} + "Std";
                features_local_flat(4*i-1) = features_local{i} + "Skew";
                features_local_flat(4*i)   = features_local{i} + "Kurt";
            end
            
            table_features_local_flat = array2table(array_features_local_summary, ...
                'VariableNames', features_local_flat);
            
            % Add labels/metadata
            table_features_local_flat.Strain = strain;
            table_features_local_flat.Record = string(csvFiles(k).name);
    
            % Put metadata first
            table_features_local_flat = movevars(table_features_local_flat, ["Strain","Record"], 'Before', 1);
            
            table_features_local_flat(:,3:end) = fillmissing(table_features_local_flat(:,3:end),"constant",0);

            dataset_all_summary{j} = [dataset_all_summary{j}; table_features_local_flat];

            % dataset_all_summary_array(j) = [dataset_all_summary_array(j,:); table_features_local_flat];
        end
        dataset_frames = [dataset_frames; table(nframes, 'VariableNames', {'NumberOfFrames'})];
        dataset_fps = [dataset_fps; table(fps, 'VariableNames', {'FramesPerSecond'})];
        dataset_strain = [dataset_strain; table(strain, 'VariableNames', {'Strain'})];
    end
end

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

for j = 1:nskip:nsummaries
    T = dataset_all_summary{j};
    outPath = fullfile(outputDir, sprintf("openworm_dataset_summary_%d.csv", j));
    writetable(T, outPath);
end

outPath = fullfile(outputDir, "openworm_dataset_frames.csv");
writetable(dataset_frames, outPath);

outPath = fullfile(outputDir, "openworm_dataset_fps.csv");
writetable(dataset_fps, outPath);

outPath = fullfile(outputDir, "openworm_dataset_strain.csv");
writetable(dataset_strain, outPath);

outMatPath = fullfile(outputDir, "openworm_dataset_time.mat");
save(outMatPath, 'dataset_all_time');

fprintf("Saved time dataset to: %s\n", outMatPath);
