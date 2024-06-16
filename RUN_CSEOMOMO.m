% Run CSEO-MOMO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: QinnaZhu,Haibo Yua, Li Kangb, Jianchao Zeng
%A Q-Learning Driven Competitive Surrogate Assisted Evolutionary Optimizer with Multiple Oriented Mutation Operators for Expensive Problems
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ gsamp1 ,time_cost] = RUN_CSEOMOMO(runs, D, FUN, LB, UB, fname)
time_begin = tic;
for r = 1:runs
    % main loop
    fprintf('\n');
    disp(['Fname:', fname,'  Run:', num2str(r)]);  
    fprintf('\n');
    [celue_save,hisx,hisf,fitcount,mf,CE,gfs] =CSEO_MOMO(FUN,D,LB,UB); 
    fprintf('Best fitness: %e\n',min(hisf)); 
    gsamp1(r,:) = gfs(1:mf);  
end
%%%%%%%%%%%%%%%%%%%%% Output options %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
best_samp   = min(gsamp1(:,end));
worst_samp  = max(gsamp1(:,end));
samp_mean   = mean(gsamp1(:,end));
samp_median = median(gsamp1(:,end));
std_samp    = std(gsamp1(:,end));
out1        = [best_samp,worst_samp,samp_mean,samp_median,std_samp];
gsamp1_ave  = mean(gsamp1,1);
gsamp1_log  = log10(gsamp1_ave);   
gsamplog    = log10(gsamp1);  

%%plot
figure(1)   %画第几个函数图 fun_num
y(:,1)=mean(gsamp1,1);  %按列求均值
Max_FES=1000
xx=1:round(Max_FES/10):Max_FES;
xx(11)=Max_FES;
tempy=log10(y(:,1));
p1=plot(xx,tempy(xx),'^-','Color',[1,0,0]);
Dim=num2str(D);


title1=strcat(fname,Dim,'D_收敛曲线')
title(title1)



xlabel('评价次数');
ylabel('适应度函数值（log）');
% fig_name=strcat('NFE',num2str(mf),'_',fname,' runs=',num2str(runs),' Dim=',num2str(D),'D_收敛曲线')
% title(fig_name)

% Time Complexity
time_cost = toc(time_begin);
time_cost = time_cost/runs;
save result
%save(strcat('E:\\研究生\\课题\\专利\\代码\\RL_DSAEA_3\\result\\30D_onlyDEcunrrentbest\\','NFE',num2str(mf),'_',fname,' runs=',num2str(runs),' Dim=',num2str(D)));
save(strcat('F:\博士生\课题\CSEO-MOMO(share)\result\','NFE',num2str(mf),'_',fname,' runs=',num2str(runs),' Dim=',num2str(D)));
path=strcat('F:\博士生\课题\CSEO-MOMO(share)\result\',title1,'.fig');
savefig(path) 
end
