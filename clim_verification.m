clear; clc; close all;
% 1. parameters
addpath(genpath('/public/home/achwjznh4b/install/toolbox/mexcdf/'));
clim_path =('/public/home/achwjznh4b/ERA5/Climatology/');
contestant_clim_path=('/public/home/pan2174/sdp/ERA5/Climatology/');
save_path = '/public/home/pan2174/sdp/verification/';
mkdir(save_path); 
rmse_clim = nan(1,92);
rmse_P90 = nan(1,92);
day_index = 1;

for month = 6:8
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31];
    for day = 1:days_in_month(month)
        filename = sprintf('%02d%02d.nc', month, day);        
        fprintf('正在读取：%s\n', filename);  
        clim_file = [clim_path,filename];
        clim_data = ncread(clim_file, 'Climmean');  
        P90_data = ncread(clim_file, 'P90_sst'); 
        contestant_file=[contestant_clim_path,filename];
        contestant_data = ncread(contestant_file, 'Climmean'); 
        contestant_P90= ncread(contestant_file, 'P90_sst'); 
        rmse_clim(day_index) = sqrt(nanmean((clim_data(:) - contestant_data(:)).^2));
        rmse_P90(day_index) =  sqrt(nanmean((P90_data(:) - contestant_P90(:)).^2));
        day_index = day_index + 1;
    end
end
R_clim=nanmean(rmse_clim);
R_P90=nanmean(rmse_P90);
max_clim = max(rmse_clim);  % 气候态最大日误差
max_P90  = max(rmse_P90);   % P90最大日误差
% ===================== 打印到屏幕 =====================
fprintf('=========================================\n');
fprintf('       6~8月夏季 RMSE 统计结果\n');
fprintf('=========================================\n');
fprintf('气候态 平均误差 : %.4f ℃\n', R_clim);
fprintf('气候态 最大误差 : %.4f ℃\n', max_clim);
fprintf('-----------------------------------------\n');
fprintf('P90分位 平均误差 : %.4f ℃\n', R_P90);
fprintf('P90分位 最大误差 : %.4f ℃\n', max_P90);
fprintf('=========================================\n');

file_clim = fullfile(save_path, 'RMSE_clim.txt');
file_P90  = fullfile(save_path, 'RMSE_P90.txt');
dlmwrite(file_clim, R_clim, 'precision', '%.4f');
dlmwrite(file_P90, R_P90, 'precision', '%.4f');
