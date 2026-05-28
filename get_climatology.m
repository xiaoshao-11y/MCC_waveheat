%% ===================== 关键参数设置（根据你的数据修改！）=====================
clear; clc; close all;
% 1. 文件路径与命名规则
addpath(genpath('/public/home/achwjznh4b/install/toolbox/mexcdf/')); %添加toolbox路径

nc_path='/public/home/achwjznh4b/Newdata/';

% 2. 时间范围与气候态天数
start_yr = 1991; end_yr = 2020;      % 气候态统计年份（30年）
n_days = 365;                        % 一年365天（无2月29）
% 3. NC文件变量名
var_sst = 'data';            % OSTIA海温变量名（标准为analysed_sst）
var_lon = 'lon';                     % 经度变量名
var_lat = 'lat';                     % 纬度变量名
var_time = 'time';                   % 时间变量名
% 4. 其他参数
delta_day = 5;                       % 前后5天，共11天（delta_day*2+1）
save_path = '/public/home/pan2174/ERA5/Climatology/';% 气候态结果保存路径（需提前创建）
if ~exist(save_path, 'dir'), mkdir(save_path); end

row_total = 721;   % 第一维：行
col_total = 1440;   % 第二维：列
dep_total = 330;    % 第三维：深度（全程完整，不拆分）
row_block = 1;   % 行方向块大小
n_row = row_total / row_block;  % 行分块数


%% ===================== 步骤1：生成365天的年日序（dayofyear）对应的datetime =====================
% 生成无2月29的365天基准日期（以2020年为基准，2020是闰年，剔除2.29）
base_yr = 2020;
base_dates = datetime(base_yr,1,1):datetime(base_yr,12,31);

%% ===================== 步骤2：预读取单个NC文件，获取经纬度和数据维度 =====================
% 读取一个示例文件，确定经纬度和海温维度
demo_nc = [nc_path, '19910101'];
if ~exist(demo_nc, 'file'), error('The NC file does not exist. Please check!'); end
lon = ncread(demo_nc, var_lon);       
lat = ncread(demo_nc, var_lat);      
lon_num = length(lon); lat_num = length(lat);

%% ===================== 步骤3：批量处理365天，逐天计算气候态及90%分位数 =====================
for doy =152:243
    fprintf('Calculating the climatological sst for the %d day...\n', doy);
    % 遍历1991~2020年，逐年份匹配日期范围
    Clim = nan(lat_num,lon_num);
    P10 = nan(lat_num,lon_num);
    P90 = nan(lat_num,lon_num);
    for i = 1:n_row 
        sst_temp = nan(lon_num,330);
        meanSST = nan(lon_num,1);
        temp_idx = 1;  
        for yr = start_yr:end_yr
            base_dates = datetime(yr,1,1):datetime(yr,12,31);
            is_leap = (mod(yr,4)==0 && mod(yr,100)~=0) || mod(yr,400)==0;
            if is_leap
                base_dates = base_dates(day(base_dates,'dayofyear')~=60); % 剔除2月29日（年日序60）
            end
            current_date = base_dates(doy);  % 当天的基准日期（2020年）
            % 计算当天的前后5天，共11天的日期范围
            date_range = current_date - days(delta_day) : current_date + days(delta_day);
            date_range = date_range';  % 转为列向量，11天
            % 初始化临时数组：存储30年中该11天的所有海温数据 [lon, lat, 30年*11天]
            % 遍历11天的日期范围，逐天读取NC文件
            for d = 1:length(date_range)
                % 将基准日期的月/日匹配到当前年份，生成待读取的日期
                target_date =date_range;
                % 转换为YYYYMMDD格式，拼接NC文件名
                date_str = datestr(target_date, 'yyyymmdd');
                nc_file = [nc_path,date_str(d,:)];
                % 检查文件是否存在，不存在则跳过并提示
                if ~exist(nc_file, 'file')
                    fprintf('warning：%s does not exist，skip this file！\n', nc_file);
                    temp_idx = temp_idx + 1;
                end
                
                % 读取海温数据
                sst = nc_varget(nc_file, var_sst,[i-1 0],[1 1440]);
                if size(sst,1) == length(lat) && size(sst,2) == length(lon)
                    sst = sst';
                end
                % 存入临时数组
                sst_temp(:,temp_idx) = sst;
                temp_idx = temp_idx + 1;
            end
        end
        n = sum(~isnan(sst_temp(:)), 1);
        if ~n==0
            meanSST= nanmean(sst_temp, 2);
            % 计算10%和90%分位数,写回结果数组对应位置
            Y= prctile(sst_temp,90,2);
            P90(i, :)=Y';
            % 计算气候态海温，写回结果数组对应位置
            Clim(i, :) = meanSST';
        end
        fprintf('Finished %d/%d \n', i, n_row);        
    end
    
    %% 将气候态海温保存为NC文件
    % 1. 创建NC文件路径和名称
    current_date = base_dates(doy);
    date=datestr(current_date, 'yyyymmdd');
    clim_nc_name = [save_path, date(5:8),'.nc'];
    if exist(clim_nc_name, 'file') == 2
        delete(clim_nc_name);
    end
    ncid = netcdf.create(clim_nc_name, 'NETCDF4'); 
    % 2. 定义维度（lat维度长度3600，lon维度长度7200）
    lat_dimid = netcdf.defDim(ncid, 'Lat', lat_num);
    lon_dimid = netcdf.defDim(ncid, 'Lon', lon_num);
    time_dimid = netcdf.defDim(ncid, 'Day',1);
    % 3. 创建变量
    time_varid = netcdf.defVar(ncid, 'dayofyear', 'double', time_dimid);
    netcdf.putAtt(ncid, time_varid, 'long_name', 'Day of year (1-365, no 29Feb)'); 
    
    lat_varid = netcdf.defVar(ncid, 'Lat','double',[lat_dimid]);    
    lon_varid = netcdf.defVar(ncid, 'Lon','double',[lon_dimid]);
    
    Clim_varid = netcdf.defVar(ncid, 'Climmean','double',[lon_dimid,lat_dimid]);
    netcdf.putAtt(ncid, Clim_varid, 'long_name', 'OSTIA SST climatology 1991-2020'); 
    
    P90_varid = netcdf.defVar(ncid, 'P90_sst','double',[lon_dimid,lat_dimid]);   
    netcdf.putAtt(ncid, P90_varid, 'long_name', '90th percentile of precipitation'); 
    
    % 4. 完成netCDF文件定义模式
    netcdf.endDef(ncid)
    
    % 5. 将数据写入netcdf文件
    netcdf.putVar(ncid, lat_varid, lat);   % 写入纬度数据
    netcdf.putVar(ncid, lon_varid, lon);   % 写入经度数据
    netcdf.putVar(ncid, time_varid, doy);
    netcdf.putVar(ncid, Clim_varid, Clim');
    netcdf.putVar(ncid, P90_varid, P90');   % 写入P90数据

    % 6 关闭文件
    netcdf.close(ncid);   
    
    disp([clim_nc_name ' was created successfully.']);
end
