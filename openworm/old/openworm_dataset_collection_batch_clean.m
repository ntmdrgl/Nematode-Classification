close all
clear
clc

% ------ USE THIS FILE TO RUN BATCH PROCESSING OF .hdf5 TO .csv
% Requirements: 
% --- Image Processing Toolkit
% --- Parallel Computing Toolkit (optional)

% --- Put base directory as folder "openworm"
baseDir = "C:\Github\CSCI-198-Nematode-Classification\openworm";

% --- Put end directory as folder "results"
resultsDir = "C:\Github\CSCI-198-Nematode-Classification\openworm\csv";

% --- SET MAX FILES - default (inf)
max_files = inf;

% --- SET MAX FRAMES - default (inf)
max_frames = inf;

% --- Set Progress Bar Update Frequency (frames) - default (1000)
progress_rate = 1000;

% --- MAKE SURE THESE ARE FILE NAMES IN OPENWORM FOLDER
strains = ["n2_herma","n2_male","cb5","aq2947","ow953","cb1112"];

filenames = strings(0,1);
filenames_features = strings(0,1);
record_names = strings(0,1);
strain_names = strings(0,1);

for s = 1:numel(strains)
    recordDirs = dir(fullfile(baseDir, strains(s), "record_*"));
    recordDirs = recordDirs([recordDirs.isdir]);

    for r = 1:numel(recordDirs)
        recName = string(recordDirs(r).name);              
        recPath = fullfile(recordDirs(r).folder, recName);

        % ---- SKIP IF ALREADY PROCESSED ----
        outCsv = fullfile(resultsDir, strains(s), recName + "_time.csv");
        if exist(outCsv, "file")
            fprintf("Skipping (already exists): %s\n", outCsv);
            continue
        end

        h5 = dir(fullfile(recPath, "*.hdf5"));

        names_all = string({h5.name});
        
        % files not ending in _features.hdf5
        keep_normal = ~endsWith(names_all, "_features.hdf5", "IgnoreCase", true);
        names_normal = names_all(keep_normal);
        
        % files ending in _features.hdf5
        keep_features = endsWith(names_all, "_features.hdf5", "IgnoreCase", true);
        names_features = names_all(keep_features);

        if ~isempty(names_normal)
            filenames(end+1,1)      = fullfile(recPath, names_normal(1));
            record_names(end+1,1)  = recName;
            strain_names(end+1,1)  = strains(s);
        end

        if ~isempty(names_features)
            filenames_features(end+1,1)      = fullfile(recPath, names_features(1));
        end 
    end
end

disp(table(strain_names, record_names, filenames));

% NOTE: Change "parfor" to "for" to not use parrallel processing toolbook
parfor k = 1:min(numel(filenames), max_files)
    fprintf('Started writing (%3d/%-3d): %s - %s\n', ...
        k, min(numel(filenames), max_files), ...
        strain_names(k), record_names(k));

    % --- read file, preprocess info
    % pixels to microns
    um_per_px = abs(double(h5readatt(filenames(k), '/mask', 'microns_per_pixel')));
    fps = h5readatt(filenames(k), '/mask', 'fps');
    
    % frame times
    t_frame = h5read(filenames(k), '/timestamp/time');
    t_frame = double(t_frame(:));
    
    % stage log
    stage_log = h5read(filenames(k), '/stage_log');  
    stage_log_t = double([stage_log.stage_time(:)].');  
    stage_log_x = double([stage_log.stage_x(:)].');
    stage_log_y = double([stage_log.stage_y(:)].');
    
    % make time unique (average x/y for duplicates)
    [t_unique, ~, g] = unique(stage_log_t, 'stable');
    x_unique = accumarray(g(:), stage_log_x(:), [], @mean);
    y_unique = accumarray(g(:), stage_log_y(:), [], @mean);
    
    % interpolate stage position to each frame time
    if numel(t_unique) >= 2
        % Normal interpolation
        stage_frame_x = interp1(t_unique, x_unique, t_frame, 'pchip', 'extrap');
        stage_frame_y = interp1(t_unique, y_unique, t_frame, 'pchip', 'extrap');
    elseif numel(t_unique) == 1
        % Only one stage measurement → assume constant position
        stage_frame_x = repmat(x_unique, size(t_frame));
        stage_frame_y = repmat(y_unique, size(t_frame));
    else
        % No valid stage data
        stage_frame_x = NaN(size(t_frame));
        stage_frame_y = NaN(size(t_frame));
    end
    
    % read features file
    skeletons = h5read(filenames_features(k), '/coordinates/skeletons');
    dorsal_contours = h5read(filenames_features(k), '/coordinates/dorsal_contours');
    ventral_contours = h5read(filenames_features(k), '/coordinates/ventral_contours');
    [~, num_points, ~] = size(skeletons);
    
    % mask size info 
    maskInfo = h5info(filenames(k), '/mask');
    num_frames = maskInfo.Dataspace.Size(3);
    
    num_features = 23;
    num_stats = 8;
    
    features_time = NaN(num_frames, num_features); % features by frames
    feature_stats = NaN(num_stats, num_features); % stats of features
    
    prev_centroid = NaN(1,2);
    prev_y_end = NaN;
    prev_x_end = NaN;

    end_swap = false;
    
    % start runtime timer
    t_start = tic;   
    
    % --- loop frames 
    for i = 1:min(num_frames, max_frames)
        % read current frame
        frame = h5read(filenames(k), '/mask', [1 1 i], [inf inf 1]);
        frame = rot90(frame);
    
        img=im2gray(frame); % grayscale the image
        [img_height,img_width]=size(img); % Image size (rows, cols)
        imgc = adapthisteq(img); % enhance contrast
        level = graythresh(imgc);      % Otsu threshold
        level = level + .10;          % change threshold
        imgb  = imbinarize(imgc, level); % binarize image
        imgb  = ~imgb; % flip white and black
        imgb = imclearborder(imgb); % remove border
    
        imgb = imclose(imgb, strel('disk', 3)); % close image with disk
        imgb = imfill(imgb, 'holes'); % fill holes
    
        cc4_imgb = bwconncomp(imgb,4); %4-connectivity
    
        % image center
        image_center_x = (img_width + 1)/2;
        image_center_y = (img_height + 1)/2;
    
        % threshold to find the worm in the image
        maxpixel=10000; % max number of pixels for the worm
        minpixel=1000; % min number of pixels for the worm

        pixLists = cc4_imgb.PixelIdxList;
        sizes = cellfun(@numel, pixLists); % size of each blob
        keep_normal = (sizes > minpixel) & (sizes < maxpixel); % logical mask
        newc = pixLists(keep_normal);
        newcsize = numel(newc);
    
        % if no blobs pass size filter, error
        if newcsize < 1
            % warning('No blob passed size filter at frame %d', i);
            continue
        end
    
        % If more than 1 blob, choose the closest blob to target_xy
        if newcsize > 1
            % compute centroids for each candidate blob 
            blob_distance = inf(newcsize,1); % squared distances
            for blob = 1:newcsize
                pix = newc{blob};
                [blob_ys, blob_xs] = ind2sub(size(imgb), pix);
                blob_x = mean(blob_xs);
                blob_y = mean(blob_ys);
                % blob_distance(blob) = hypot((blob_x - target_x), (blob_y - target_y));
                blob_distance(blob) = hypot((blob_x - image_center_x), (blob_y - image_center_y));
            end
        
            [~, bestIdx] = min(blob_distance);
            newc = { newc{bestIdx} };   % keep only the chosen blob
        end
    
        area=size(newc{1},1); % chooses the worm blob
    
        vpixels=zeros(area,2);
        [vpixels(:,1), vpixels(:,2)] = ind2sub(size(imgb), newc{1});
    
        % Find bounding box to crop image
        imin=min(vpixels(:,1));
        imax=max(vpixels(:,1));
        jmin=min(vpixels(:,2));
        jmax=max(vpixels(:,2));
        
        ROI_width=imax-imin+1;
        ROI_height=jmax-jmin+1;
    
        ROI = false(ROI_width, ROI_height);
        r = vpixels(:,1) - imin + 1;
        c = vpixels(:,2) - jmin + 1;
        ROI(sub2ind(size(ROI), r, c)) = true;
    
        % stats gets shape properties of the blob
        stats = regionprops(logical(ROI), ...
                            'Centroid', ...
                            'Orientation', ...
                            'Perimeter' ...
                            );
    
        % rotate ROI is worm is straight according to orientation
        ROIrot = imrotate(ROI,-stats.Orientation);
    
        col_counts = sum(ROIrot~=0, 1); % non zero elements per column
        col_counts_nz = col_counts(col_counts > 0); % at least one element per column
    
        width_mean = mean(col_counts_nz);
    
        % Measure bounding box after rotation
        stats_rot = regionprops(logical(ROIrot), 'BoundingBox');
    
        ROIrot_width = stats_rot.BoundingBox(3);
        ROIrot_height = stats_rot.BoundingBox(4);
        aspect_ratio = ROIrot_width/ROIrot_height;
    
        % Skeletionize to 1-pixel wide spine
        skel = bwskel(logical(ROI));
        skel_length = nnz(skel);
        
        % Find endpoints of skel using 8 connectivity
        endpts = bwmorph(skel, 'endpoints');
        %endpts = bwmorph(endpts, 'spur', 5);
        %endpts = bwmorph(endpts, 'clean');
        endpoint_num = nnz(endpts); % number of endpoints found
        
        [end_y, end_x] = find(endpts); % find local non-zero coords of endpts
        
        endpoint_length = NaN;
        end_global_x = NaN(1,2);
        end_global_y = NaN(1,2);
        endpoint_displacement_1 = NaN;
        endpoint_displacement_2 = NaN;
    
        endpoint_x1_um = NaN;
        endpoint_x2_um = NaN;
        endpoint_y1_um = NaN;
        endpoint_y2_um = NaN;
    
        bends_num = NaN;
        bends_length_mean = NaN;
        bends_length_std = NaN;
    
        % have exactly two endpoints
        if numel(end_y) == 2      
            % uses local endpoints
            endpoint_length = hypot(end_x(1) - end_x(2), ...
                                      end_y(1) - end_y(2));
    
            % find global endpoints
            end_global_x = end_x + (jmin - 1);
            end_global_y = end_y + (imin - 1);
            
            %
            % draw a line local endpoints
            x1 = end_x(1); y1 = end_y(1);
            x2 = end_x(2); y2 = end_y(2);
            
            num_samples = max(abs(x2-x1), abs(y2-y1)) + 1;
            xs = round(linspace(x1, x2, num_samples));
            ys = round(linspace(y1, y2, num_samples));
            
            % clip to image bounds
            xs = min(max(xs, 1), size(skel,2));
            ys = min(max(ys, 1), size(skel,1));
            
            lineMask = false(size(skel));
            lineMask(sub2ind(size(skel), ys, xs)) = true;
            
            lineOnSkel = lineMask & skel;
            lineOnSkel = imdilate(lineOnSkel, strel('disk', 3));
            %}
    
            % --- Centroids of each connected blob in lineOnSkel ---
            cc8_lineOnSkel = bwconncomp(lineOnSkel, 8); 
            stats_lineOnSkel = regionprops(cc8_lineOnSkel, 'Centroid');
            
            bends_centroid = vertcat(stats_lineOnSkel.Centroid);
    
            bends_num = height(bends_centroid) - 1;
    
            bends_diff_x = diff(bends_centroid(:,1));
            bends_diff_y = diff(bends_centroid(:,2));
        
            bends_lengths = hypot(bends_diff_x, bends_diff_y);
        
            bends_length_mean = mean(bends_lengths);
            bends_length_std  = std(bends_lengths);

            % get endpoints from skeletons
            if (i <= size(skeletons,3))
                endpoint_x1_um = skeletons(1, 1, i);
                endpoint_y1_um = skeletons(2, 1, i);
        
                endpoint_x2_um = skeletons(1, num_points, i);
                endpoint_y2_um = skeletons(2, num_points, i);
            end
            
            % first frame displacement
            if (i == 1)
                endpoint_displacement_1 = 0;
                endpoint_displacement_2 = 0;
            end
    
            end_swap = false;
            % find minimum distance moved from either the tail or head
            if numel(prev_y_end) == 2
                %
                % first case, endpoints don't swap
                d11 = hypot(endpoint_x1_um-prev_x_end(1), endpoint_y1_um-prev_y_end(1));
                d22 = hypot(endpoint_x2_um-prev_x_end(2), endpoint_y2_um-prev_y_end(2));
                
                % second case, endpoints swap
                d12 = hypot(endpoint_x1_um-prev_x_end(2), endpoint_y1_um-prev_y_end(2));
                d21 = hypot(endpoint_x2_um-prev_x_end(1), endpoint_y2_um-prev_y_end(1));
    
                sum_no_swap = d11 + d22;
                sum_swap = d12 + d21;
                
                % minimize displacement of endpoints
                if sum_no_swap <= sum_swap
                    endpoint_displacement_1 = d11;
                    endpoint_displacement_2 = d22;
                else
                    endpoint_displacement_1 = d12;
                    endpoint_displacement_2 = d21;
                    end_swap = true;
                end
                %}
            end
    
            if (end_swap == true)
                % must swap endpoints
                endpoint_temp_x = endpoint_x1_um;
                endpoint_temp_y = endpoint_y1_um;
    
                endpoint_x1_um = endpoint_x2_um;
                endpoint_y1_um = endpoint_y2_um;
    
                endpoint_x2_um = endpoint_temp_x;
                endpoint_y2_um = endpoint_temp_y;
            end
    
            if (~isnan(endpoint_x1_um) && ~isnan(endpoint_x2_um) ...
                && ~isnan(endpoint_y1_um) && ~isnan(endpoint_y2_um))
    
                prev_x_end = [endpoint_x1_um; endpoint_x2_um];
                prev_y_end = [endpoint_y1_um; endpoint_y2_um];
            end
        end
        centroid_x_um = NaN;
        centroid_y_um = NaN;
        
        % Centroid is mean of all coordinates
        if (i <= size(skeletons,3))
            centroid_x_um = mean([skeletons(1, :, i), dorsal_contours(1, :, i), ventral_contours(1, :, i)]);
            centroid_y_um = mean([skeletons(2, :, i), dorsal_contours(2, :, i), ventral_contours(2, :, i)]);
        end
        
    
        % displacement
        if i == 1
            centroid_displacement = 0; % first frame distance is 0
        else
            % euclidean distance between current and previous frame
            centroid_displacement = hypot(prev_centroid(1) - centroid_x_um, ...
                                  prev_centroid(2) - centroid_y_um);
        end
        
        if (~isnan(centroid_x_um) && ~isnan(centroid_y_um))
            prev_centroid = [centroid_x_um, centroid_y_um]; 
        end
    
        angle = stats.Orientation;
        perimeter = stats.Perimeter;
    
        tortuosity = skel_length / endpoint_length;
    
        features_time(i, :) = [
            skel_length, ... 
            centroid_x_um, ...
            centroid_y_um, ...
            centroid_displacement, ...
            endpoint_num, ...
            endpoint_length, ...
            endpoint_x1_um, ...
            endpoint_x2_um, ...
            endpoint_y1_um, ...
            endpoint_y2_um, ...
            endpoint_displacement_1, ...
            endpoint_displacement_2, ...
            end_swap, ...
            width_mean, ...
            aspect_ratio, ...
            area, ...
            perimeter, ...
            angle, ...
            tortuosity, ...
            bends_num, ...
            bends_length_mean, ...
            bends_length_std, ...
            fps ...
        ];
    
        if (mod(i,progress_rate) == 0 || i == 1)
            elapsed = toc(t_start); 
            rate = elapsed / i; 
            remaining = rate * (min(num_frames, max_frames) - i); 
            
            finish_time = datetime('now') + seconds(remaining);
        
            fprintf(['\tProgress (%3d/%-3d): %.1f%% (%d/%d) | ' ...
                     'Elapsed: %s | Remaining: %s\n'], ...
                     k, min(numel(filenames), max_files), ...
                     100*i/min(num_frames, max_frames), i, min(num_frames, max_frames), ...
                     char(duration(0,0,elapsed,'Format','hh:mm:ss')), ...
                     char(duration(0,0,remaining,'Format','hh:mm:ss')) ...
                     )
        end
    end
    
    feature_names = {
        'Skeleton length'
        'Centroid X'
        'Centroid Y'
        'Centroid displacement'
        'Number of endpoints'
        'Endpoint separation'
        'Endpoint X1'
        'Endpoint X2'
        'Endpoint Y1'
        'Endpoint Y2'
        'Endpoint displacement 1'
        'Endpoint displacement 2'
        'Endpoint swap'
        'Width mean'
        'Aspect ratio'
        'Area'
        'Perimeter'
        'Orientation angle'
        'Tortuosity'
        'Number of bends'
        'Bend length mean'
        'Bend length std'
        'FPS'
    };
    
    % --- features over time (frames × features) ---
    features_time_table = array2table( ...
        features_time, ...
        'VariableNames', feature_names ...
    );
    
    writetable(features_time_table, "openworm\csv\" + strain_names(k) + "\" + record_names(k) + "_time.csv");

    fprintf('\t\tFinished writing (%3d/%-3d): %s - %s\n\n', k, min(numel(filenames), max_files), strain_names(k), record_names(k));
end

fprintf('That s all folks\n');