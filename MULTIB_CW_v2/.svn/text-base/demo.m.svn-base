

% code author: Guosheng Lin
% contact: guosheng.lin@gmail.com or guosheng.lin@adelaide.edu.au

% this code is for the following paper, please cite:
% [1] Guosheng Lin, Chunhua Shen, Anton van den Hengel and David Suter;
% "Fast training of effective multi-class boosting using coordinate descent optimization", ACCV2012.

% note that this code is an improved and simplified version of our accv paper.
 

% this code also use the following code for decision tree implementation:
% Piotr's Image & Video Matlab Toolbox
% http://vision.ucsd.edu/~pdollar/toolbox/doc/
% please cite their paper:
% [2] R. Appel, T. Fuchs, P. Dollár, P. Perona; 
% "Quickly Boosting Decision Trees – Pruning Underachieving Features Early," ICML 2013.




addpath(genpath([pwd '/piotr_toolbox/']));
addpath(genpath([pwd '/multib_cw/']));

% try these two dataset:
ds_file_name='uci_vowel';
% ds_file_name='uci_usps';

% load the demo dataset:
dataset_stru=load(['./dataset_mc/' ds_file_name '.mat']);
ds=dataset_stru.dataset;

% here quantize the data into int8.
ds.x=quantize_data(ds.x);


% this is a very simple way to split training data, this is not the code used in
% the paper.

e_num=length(ds.y);
trn_e_num=round(e_num*0.7);

trn_sel=false(e_num,1);
trn_sel(randsample(e_num,trn_e_num))=true;
ds.train_inds=find(trn_sel);
ds.test_inds=find(~trn_sel);

train_data=[];
train_data.feat_data=ds.x(ds.train_inds,:);
train_data.label_data=ds.y(ds.train_inds);

test_data=[];
test_data.feat_data=ds.x(ds.test_inds,:);
test_data.label_data=ds.y(ds.test_inds);




% boosting iteration setting:
max_iteration_num=100;
% max_iteration_num=200;
% max_iteration_num=500;


% decision tree setting: large depth may converge faster, but may overfit
pTree=[];
pTree.maxDepth=2;
pTree.nThreads=4;
% pTree.fracFtrs=0.5;


% for performance evaluation:
eva_step=10;


% 3 choices for running, see the comments below
method_type=1;


if method_type==1
% 1. Multiboost-CW, this is the fastest way, stagewise setting:
    train_info=[];
    train_info.notes='MultiB-CW-Stagewise';
    train_info.use_stagewise=true;
end


if method_type==2
% 2. Multiboost-CW, this will get lower training error, totoal corrective setting:
    train_info=[];
    train_info.notes='MultiB-CW';
    train_info.use_stagewise=false;
    train_info.max_ws_iter=2;
    train_info.tradeoff_nv=1e-5;
end

if method_type==3
% 3. trun off the Multiboost-CW, back to the MultiBoost formulation, 
% but still use the fast codinate decent solver proposed in our paper.
% this will converge slower.
    train_info=[];
    train_info.notes='MultiB';
    train_info.use_cw=false;
    train_info.use_stagewise=false;
    train_info.max_ws_iter=2;
    train_info.tradeoff_nv=1e-5;
end



train_info.max_iteration_num=max_iteration_num;
train_info.train_data=train_data;
train_info.pTree=pTree;
train_info.notes=[train_info.notes '-tree-depth-' num2str(pTree.maxDepth)];
train_info.train_id=[ds_file_name '_' train_info.notes];

train_result=multibcw_learn(train_info);

predict_config=[];
predict_config.eva_iter_idxes=eva_step:eva_step:train_info.max_iteration_num;
predict_result=multibcw_predict(train_result.model, test_data, predict_config);


disp('train_result:');
disp(train_result);
disp('predict_result');
disp(predict_result);



f1=figure;
line_width=2;
xy_font_size=22;
marker_size=10;
legend_font_size=10;
xy_v_font_size=15;
title_font_size=xy_font_size;

color=gen_color(1);
marker=gen_marker(1);

figure(f1);
p=plot(predict_result.eva_iter_idxes, predict_result.error_rate_iters);
set(p,'Color', color)
set(p,'Marker',marker);
set(p,'LineWidth',line_width);
set(p,'MarkerSize',marker_size);

legend_strs_f1=[train_result.model.notes ' (' num2str(predict_result.error_rate_iters(end)) ')'];

hleg=legend(legend_strs_f1);
% set(hleg, 'FontSize',legend_font_size);
set(hleg,'Location','NorthEast');
grid on;
title('error rate', 'FontSize', title_font_size);
h1=xlabel('iterations');
h2=ylabel('error rate');
set(h1, 'FontSize',xy_font_size);
set(h2, 'FontSize',xy_font_size);
set(gca, 'FontSize',xy_v_font_size);
set(hleg, 'FontSize',legend_font_size);


