function openworm_matlab_features_new(videoFile, outputFile, maxFrames, readBlockFrames, frameWorkers, poolType)
% Extract the 16 features selected by the original preprocessing script.
% Requires MATLAB Image Processing Toolbox. Example:
% openworm_matlab_features('worm.hdf5', 'features.mat', 100)
% Revised version: timestamp alignment, physical units, and unscaled net
% displacement. Segmentation, skeletonization and bend definitions are retained.
% Lengths: micrometers; area: micrometers^2; speeds: micrometers/second;
% orientation: degrees; counts and ratios: dimensionless.
% Optional frameWorkers (5th argument, default 1) uses bounded parfor batches
% for independent image calculations only. Tracking and feature assembly remain
% in chronological order. poolType (6th argument) defaults to 'Threads', which
% shares memory; 'Processes' is also supported. Requires Parallel Computing
% Toolbox when frameWorkers > 1. Image formulas and arithmetic are unchanged.
% SkeletonLength: endpoint-to-endpoint geodesic on the mask skeleton, using
% Freeman 8-direction chain-link weights (axial=1, diagonal=sqrt(2)), then um/px.
% H. Freeman (1961), On the Encoding of Arbitrary Geometric Configurations,
% IRE Trans. Electronic Computers EC-10(2):260-268. doi:10.1109/TEC.1961.5219197.
% Explicit link-length formula: Freeman (1974), Computer Processing of
% Line-Drawing Images, ACM Computing Surveys 6(1):57-97, Eq. (5), p. 76.
% doi:10.1145/356625.356627.
% Geodesic implementation: bwdistgeodesic(...,'quasi-euclidean'); see
% https://www.mathworks.com/help/images/ref/bwdistgeodesic.html (Soille, 2004).
% This measures the digital centerline, not the perimeter of a closed contour;
% no closed-boundary corner correction or endpoint extension is applied.
% The companion *_features.hdf5 supplies coordinates, not features_timeseries.
% Ignore trailing timestamp entries without mask images and repeated identical
% acquisition frame/time pairs. Keep original image indices for HDF5 reads;
% conflicting duplicates, missing timestamps and backwards clocks are errors.
if nargin < 3
    maxFrames = inf;
end
if nargin < 4
    readBlockFrames = 32;
end
assert(isscalar(readBlockFrames) && isfinite(readBlockFrames) && ...
    readBlockFrames >= 1 && mod(readBlockFrames,1) == 0, 'Invalid readBlockFrames');
% Batch only disk reads; frames are still calculated sequentially. A value of
% 1 disables read batching for comparisons. Cache is bounded to about 32 MiB
% (or one frame, if a single frame exceeds that budget).
if nargin < 5
    frameWorkers = 1;
end
if nargin < 6
    poolType = 'Threads';
end
assert(isscalar(frameWorkers) && isfinite(frameWorkers) && frameWorkers >= 1 && ...
    mod(frameWorkers,1) == 0, 'Invalid frameWorkers');
if frameWorkers > 1
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool(poolType, frameWorkers);
    end
    assert(pool.NumWorkers == frameWorkers, 'Existing pool has different worker count');
end
featuresFile = replace(string(videoFile), '.hdf5', '_features.hdf5');
assert(isfile(videoFile), 'Missing video HDF5 file');
assert(isfile(featuresFile), 'Missing companion features HDF5 file');
assert(maxFrames > 0 && (isinf(maxFrames) || mod(maxFrames,1) == 0), 'Invalid maxFrames');
progress_rate = 1000;
assert(exist('adapthisteq','file') ~= 0 && license('test','image_toolbox'), ...
    'openworm:MissingToolbox', 'MATLAB Image Processing Toolbox must be installed and licensed.');
    % --- read file, preprocess info
    % pixels to microns
    um_per_px = abs(double(h5readatt(videoFile, '/mask', 'microns_per_pixel')));
    fps = h5readatt(videoFile, '/mask', 'fps');
    assert(isscalar(um_per_px) && isfinite(um_per_px) && um_per_px > 0, ...
        'Invalid video microns_per_pixel');
    
    % frame times
    t_frame = h5read(videoFile, '/timestamp/time');
    t_frame = double(t_frame(:));
    
    % stage log
    stage_log = h5read(videoFile, '/stage_log');  
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
    skeletons = h5read(featuresFile, '/coordinates/skeletons');
    dorsal_contours = h5read(featuresFile, '/coordinates/dorsal_contours');
    ventral_contours = h5read(featuresFile, '/coordinates/ventral_contours');
    [~, num_points, ~] = size(skeletons);
    
    % mask size info 
    maskInfo = h5info(videoFile, '/mask');
    num_frames = maskInfo.Dataspace.Size(3);
    % Match raw acquisition frame numbers, not array row positions. Feature
    % tracks can start late, end early, or omit acquisition frames.
    video_frames = double(h5read(videoFile, '/timestamp/raw'));
    video_frames = video_frames(:);
    assert(numel(video_frames) == numel(t_frame) && numel(t_frame) >= num_frames, ...
        'Video timestamp arrays must match and cover every mask image');
    if numel(t_frame) > num_frames
        warning('openworm:TrailingTimestamps', ...
            'Ignoring %d trailing timestamp entries without mask images.', numel(t_frame)-num_frames);
        video_frames = video_frames(1:num_frames);
        t_frame = t_frame(1:num_frames);
    end
    assert(all(isfinite(video_frames)) && all(video_frames == floor(video_frames)) && ...
        all(isfinite(t_frame)), 'Video timestamps must be finite with integer frame numbers');
    duplicate_frames = [false; diff(video_frames) == 0 & diff(t_frame) == 0];
    valid_video_rows = ~duplicate_frames;
    assert(all(diff(video_frames(valid_video_rows)) > 0) && ...
        all(diff(t_frame(valid_video_rows)) > 0), ...
        'Video frame numbers and times must increase (except identical repeated pairs)');
    if any(duplicate_frames)
        warning('openworm:DuplicateTimestamps', ...
            'Ignoring %d mask images with repeated acquisition frame/time pairs.', nnz(duplicate_frames));
    end
    feature_table = h5read(featuresFile, '/features_timeseries');
    feature_frames = double(feature_table.timestamp(:));
    assert(numel(feature_frames) == size(skeletons,3) && ...
        size(dorsal_contours,3) == size(skeletons,3) && ...
        size(ventral_contours,3) == size(skeletons,3) && ...
        all(isfinite(feature_frames)) && all(diff(feature_frames) > 0) && ...
        all(feature_frames == floor(feature_frames)), 'Invalid feature frame alignment');
    worms = double(feature_table.worm_index(:));
    assert(numel(unique(worms(isfinite(worms) & worms > 0))) <= 1, ...
        'Expected a single worm track');
    [~, coordinate_rows] = ismember(video_frames, feature_frames);
    % Tierpsy coordinate units are declared on features_timeseries. Never
    % multiply already calibrated micrometer coordinates by pixel size again.
    xy_units = lower(strtrim(string(h5readatt(featuresFile, '/features_timeseries', 'xy_units'))));
    if any(xy_units == ["micrometers", "micrometres", "microns", "um"])
        coordinate_scale = 1;
    elseif any(xy_units == ["pixels", "pixel", "px"])
        coordinate_scale = abs(double(h5readatt(featuresFile, '/features_timeseries', 'microns_per_pixel')));
    else
        error('Unsupported feature coordinate units: %s', xy_units);
    end
    assert(isscalar(coordinate_scale) && isfinite(coordinate_scale) && coordinate_scale > 0, ...
        'Invalid feature coordinate calibration');
    skeletons = double(skeletons) * coordinate_scale;
    dorsal_contours = double(dorsal_contours) * coordinate_scale;
    ventral_contours = double(ventral_contours) * coordinate_scale;
    block_frames = max(1, min(readBlockFrames, floor(32 * 1024^2 / ...
        (prod(maskInfo.Dataspace.Size(1:2)) * maskInfo.Datatype.Size))));
    frame_block_start = 0;
    frame_block_end = 0;
    closing_disk = strel('disk', 3);
    
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
    for i = 1:min(num_frames, maxFrames)
        if ~valid_video_rows(i)
            continue
        end
        coordinate_row = coordinate_rows(i);
        % read current frame
        % OPTIMIZED_READ_BEGIN: preserve exact pixels, class, order and indexing.
        if i > frame_block_end
            frame_block_start = i;
            frame_block_end = min([num_frames, maxFrames, i + block_frames - 1]);
            frame_block = h5read(videoFile, '/mask', [1 1 i], ...
                [inf inf frame_block_end - frame_block_start + 1]);
            batch_count = frame_block_end - frame_block_start + 1;
            frame_results = cell(1, batch_count);
            if frameWorkers > 1
                parfor frame_index = 1:batch_count
                    frame_results{frame_index} = calculate_frame(frame_block(:,:,frame_index), closing_disk);
                end
            else
                for frame_index = 1:batch_count
                    frame_results{frame_index} = calculate_frame(frame_block(:,:,frame_index), closing_disk);
                end
            end
        end
        % OPTIMIZED_READ_END
        measurement = frame_results{i - frame_block_start + 1};
        if isempty(measurement)
            continue
        end
        area = measurement.area;
        stats = measurement.stats;
        width_mean = measurement.width_mean;
        aspect_ratio = measurement.aspect_ratio;
        skel = measurement.skel;
        skel_length = measurement.skel_length;
        endpoint_num = measurement.endpoint_num;
        end_y = measurement.end_y;
        end_x = measurement.end_x;
        imin = measurement.imin;
        jmin = measurement.jmin;
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
            lineOnSkel = imdilate(lineOnSkel, closing_disk);
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
            if coordinate_row > 0
                endpoint_x1_um = skeletons(1, 1, coordinate_row);
                endpoint_y1_um = skeletons(2, 1, coordinate_row);
        
                endpoint_x2_um = skeletons(1, num_points, coordinate_row);
                endpoint_y2_um = skeletons(2, num_points, coordinate_row);
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
        if coordinate_row > 0
            centroid_x_um = mean([skeletons(1, :, coordinate_row), dorsal_contours(1, :, coordinate_row), ventral_contours(1, :, coordinate_row)]);
            centroid_y_um = mean([skeletons(2, :, coordinate_row), dorsal_contours(2, :, coordinate_row), ventral_contours(2, :, coordinate_row)]);
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
            remaining = rate * (min(num_frames, maxFrames) - i); 
            
            finish_time = datetime('now') + seconds(remaining);
        
            fprintf(['\tProgress (%3d/%-3d): %.1f%% (%d/%d) | ' ...
                     'Elapsed: %s | Remaining: %s\n'], ...
                     1, 1, ...
                     100*i/min(num_frames, maxFrames), i, min(num_frames, maxFrames), ...
                     char(duration(0,0,elapsed,'Format','hh:mm:ss')), ...
                     char(duration(0,0,remaining,'Format','hh:mm:ss')) ...
                     )
        end
    end
    


% Apply the feature definitions selected by old/openworm_dataset_preprocessing.m.
raw_names = {'SkeletonLength','CentroidX','CentroidY','CentroidDisplacement', ...
    'NumberOfEndpoints','EndpointSeparation','EndpointX1','EndpointX2', ...
    'EndpointY1','EndpointY2','EndpointDisplacement1','EndpointDisplacement2', ...
    'EndpointSwap','WidthMean','AspectRatio','Area','Perimeter', ...
    'OrientationAngle','Tortuosity','NumberOfBends','BendLengthMean','BendLengthStd','FPS'};
n = min(num_frames,maxFrames);
output_rows = find(valid_video_rows(1:n));
t_frame = t_frame(output_rows);
video_frames = video_frames(output_rows);
n = numel(output_rows);
table_original = array2table(features_time(output_rows,:), 'VariableNames', raw_names);
% Convert image-derived quantities once; coordinate-derived columns already
% use micrometers. Ratios, counts, angles and FPS are not length quantities.
length_columns = [1 6 14 17 21 22];
table_original{:,length_columns} = table_original{:,length_columns} * um_per_px;
table_original.Area = table_original.Area * um_per_px^2;
table_filled = fillmissing(table_original,'linear');
fps = table_filled.FPS(1);
% FS0 has lag=1. The other five legacy frame-skipping lags are not selected.
coordinate_columns = {[2 3],[7 9],[8 10]};
speed_names = {'CentroidDisplacementFS0','Endpoint1DisplacementFS0','Endpoint2DisplacementFS0'};
total_names = {'CentroidTotalDisplacement','Endpoint1TotalDisplacement','Endpoint2TotalDisplacement'};
observed = false(n,6);
for point = 1:3
    columns = coordinate_columns{point};
    xy = table_filled{:,columns};
    table_filled.(speed_names{point}) = [0; hypot(diff(xy(:,1)),diff(xy(:,2))) ./ diff(t_frame(1:n))];
    table_filled.(total_names{point}) = hypot(xy(:,1)-xy(1,1),xy(:,2)-xy(1,2));
    present = all(isfinite(table_original{:,columns}),2);
    observed(:,point) = present & [true; present(1:end-1)];
    observed(:,point+3) = present;
end
table_filled.EndpointSeparation = hypot(table_filled.EndpointX1-table_filled.EndpointX2, ...
                                      table_filled.EndpointY1-table_filled.EndpointY2);
feature_names = {'SkeletonLength','EndpointSeparation','CentroidDisplacementFS0', ...
    'Endpoint1DisplacementFS0','Endpoint2DisplacementFS0','CentroidTotalDisplacement', ...
    'Endpoint1TotalDisplacement','Endpoint2TotalDisplacement','Tortuosity', ...
    'NumberOfBends','BendLengthStd','WidthMean','AspectRatio','Area','Perimeter','OrientationAngle'};
feature_values = table_filled{:,feature_names};
% Same legacy negative-value clipping, except for OrientationAngle.
negative = feature_values(:,1:15) < 0;
first_fifteen = feature_values(:,1:15);
first_fifteen(negative) = 0;
feature_values(:,1:15) = first_fifteen;
% Retain original availability so global legacy fillmissing cannot hide gaps
% from Python's stricter <0.25s interpolation/zeroing and status flags.
observed_mask = false(n,16);
observed_mask(:,1) = isfinite(table_original.SkeletonLength);
observed_mask(:,2) = all(isfinite(table_original{:,[7 8 9 10]}),2);
observed_mask(:,3:8) = observed;
raw_columns = [19 20 22 14 15 16 17 18];
observed_mask(:,9:16) = isfinite(table_original{:,raw_columns});
time_s = t_frame(1:n);
frame_number = video_frames(1:n);
feature_units = {'um','um','um/s','um/s','um/s','um','um','um','1','count', ...
    'um','um','1','um^2','um','degrees'};
save(outputFile, 'feature_values', 'feature_names', 'feature_units', 'observed_mask', 'time_s', 'frame_number', 'fps', '-v7');
end

function result = calculate_frame(frame, closing_disk)
        frame = rot90(frame);
    
        img=im2gray(frame); % grayscale the image
        [img_height,img_width]=size(img); % Image size (rows, cols)
        imgc = adapthisteq(img); % enhance contrast
        level = graythresh(imgc);      % Otsu threshold
        level = level + .10;          % change threshold
        imgb  = imbinarize(imgc, level); % binarize image
        imgb  = ~imgb; % flip white and black
        imgb = imclearborder(imgb); % remove border
    
        imgb = imclose(imgb, closing_disk); % close image with disk
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
            result = [];
            return
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
        skel_length = NaN;
        
        % Find endpoints of skel using 8 connectivity
        endpts = bwmorph(skel, 'endpoints');
        %endpts = bwmorph(endpts, 'spur', 5);
        %endpts = bwmorph(endpts, 'clean');
        endpoint_num = nnz(endpts); % number of endpoints found
        
        [end_y, end_x] = find(endpts); % find local non-zero coords of endpts
        % Measure the path within the skeleton, never the straight endpoint
        % chord. Ambiguous endpoint counts and disconnected paths stay missing.
        if numel(end_y) == 2
            skeleton_distance = bwdistgeodesic(skel, end_x(1), end_y(1), 'quasi-euclidean');
            path_length = double(skeleton_distance(end_y(2), end_x(2)));
            if isfinite(path_length) && path_length > 0
                skel_length = path_length; % pixel units; calibrated below
            end
        end
        

result = struct();
result.area = area;
result.stats = stats;
result.width_mean = width_mean;
result.aspect_ratio = aspect_ratio;
result.skel = skel;
result.skel_length = skel_length;
result.endpoint_num = endpoint_num;
result.end_y = end_y;
result.end_x = end_x;
result.imin = imin;
result.jmin = jmin;
end
