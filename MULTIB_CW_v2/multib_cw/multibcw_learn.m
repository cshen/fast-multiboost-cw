

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



function train_result=multibcw_learn(train_info)

fpraintf('\n\n #################### multibcw_learn ########################\n\n')

fpraintf('generating cache....\n');

if ~isfield(train_info, 'use_cw')
    train_info.use_cw=true;
end
use_cw=train_info.use_cw;

if ~isfield(train_info, 'use_stagewise')
    train_info.use_stagewise=true;
end

if ~isfield(train_info, 'use_solver_mex')
    train_info.use_solver_mex=true;
end



train_data=train_info.train_data;
train_data.label_vs=unique(train_data.label_data);

struct_data=do_gen_struct_data_mc(train_data, use_cw);
train_info.struct_data=struct_data;

train_info.wl_cache=gen_wl_cache(train_info);


label_vs=train_data.label_vs;
class_num=length(label_vs);
train_info.class_num=class_num;
train_info.e_num=size(train_data.feat_data,1);
train_info.feat_num=size(train_data.feat_data,2);



train_info=boost_learn_config(train_info);
train_info.struct_cache_info=struct_data.struct_cache_info;


if use_cw
    train_info.find_wlearner_fn=@find_wlearner_mc_cw;
    train_info.calc_pair_feat_fn=@calc_pair_feat_mc_cw;
    train_info.gen_cache_info_fn=@gen_cache_info_mc;
else
    train_info.find_wlearner_fn=@find_wlearner_mc;
    train_info.calc_pair_feat_fn=@calc_pair_feat_mc;
    train_info.gen_cache_info_fn=@gen_cache_info_mc;
end

train_info.solver_fn=@boost_solver_cd;

train_result=boost_learn(train_info);


model=train_result.model;
model.label_vs=label_vs;
model.class_num=class_num;
model.use_cw=use_cw;


if use_cw
    model=gen_model_mc_cw(model);
else
    model=gen_model_mc(model);
end


train_result.model=model;


fpraintf('\n\n #################### multibcw_learn finished ########################\n\n')


end




function [cache_info train_info]=gen_cache_info_mc(train_info)

struct_cache_info=train_info.struct_cache_info;
struct_data=train_info.struct_data;


sel_pair_e_left_org=struct_data.struct_pairs(:,1);

cache_info=[];
cache_info.pair_e_left=sel_pair_e_left_org;

cache_info.tensor_info=struct_cache_info.tensor_info;
cache_info.pair_num=length(sel_pair_e_left_org);
cache_info.class_num=train_info.class_num;

cg_cache=[];

class_num=train_info.class_num;
wl_e_num=struct_data.struct_e_num;
wl_pos_e_poses=struct_cache_info.e_pair_idxes;

neg_e_pair_idxes_classes=struct_cache_info.neg_e_pair_idxes_classes;


pos_e_sel_classes=cell(class_num,1);
neg_e_sel_classes=cell(class_num,1);
neg_e_poses_classes=cell(class_num,1);
wl_labels_classes=cell(class_num,1);
for c_idx=1:class_num
    one_e_idxes=struct_cache_info.cached_e_idxes_classes{c_idx};
        
    neg_e_sel=true(wl_e_num, 1);
    neg_e_sel(one_e_idxes)=false;
    pos_e_sel=~neg_e_sel;
    
    pos_e_sel_classes{c_idx}=pos_e_sel;
    neg_e_sel_classes{c_idx}=neg_e_sel;
       
    neg_e_poses_classes{c_idx}=neg_e_pair_idxes_classes{c_idx};
    
    
    wl_labels=-ones(wl_e_num,1);
    wl_labels(pos_e_sel)=1;
    wl_labels_classes{c_idx}=wl_labels;
end
cg_cache.wl_labels_classes=wl_labels_classes;
cg_cache.pos_e_sel_classes=pos_e_sel_classes;
cg_cache.neg_e_sel_classes=neg_e_sel_classes;
cg_cache.neg_e_poses_classes=neg_e_poses_classes;



cg_cache.wl_example_num=wl_e_num;
cg_cache.wl_pos_e_poses=wl_pos_e_poses;
cg_cache.pair_num=cache_info.pair_num;
cg_cache.class_num=cache_info.class_num;

cache_info.cg_cache=cg_cache;

end



function hs =find_wlearner_mc_cw(train_info, cache_info, pair_weights)


cg_cache=cache_info.cg_cache;


wl_pos_e_poses=cg_cache.wl_pos_e_poses;

wl_data_weight_mat=pair_weights(wl_pos_e_poses);
wl_data_weight_pos=sum(wl_data_weight_mat,2);

class_num=cg_cache.class_num;

hs=[];

for c_idx=1:class_num
        
    pos_e_sel=cg_cache.pos_e_sel_classes{c_idx};
    neg_e_sel=cg_cache.neg_e_sel_classes{c_idx};
    neg_e_poses=cg_cache.neg_e_poses_classes{c_idx};

    wl_data_weight=zeros(cg_cache.wl_example_num,1);
    wl_data_weight(pos_e_sel)=wl_data_weight_pos(pos_e_sel);
    wl_data_weight(neg_e_sel)=pair_weights(neg_e_poses);
    
    label_data=cg_cache.wl_labels_classes{c_idx};
    wlearner=train_wl(train_info, label_data, wl_data_weight);
     
    hs=cat(1, hs, wlearner);

end


disp_loop_info(train_info, cache_info, hs);

end



function hs=find_wlearner_mc(train_info, cache_info, pair_weights)

cg_cache=cache_info.cg_cache;

wl_pos_e_poses=cg_cache.wl_pos_e_poses;

wl_data_weight_mat=pair_weights(wl_pos_e_poses);
wl_data_weight_pos=sum(wl_data_weight_mat,2);

best_wl=[];
best_wl_score=-inf;

class_num=cg_cache.class_num;

for c_idx=1:class_num
        
    pos_e_sel=cg_cache.pos_e_sel_classes{c_idx};
    neg_e_sel=cg_cache.neg_e_sel_classes{c_idx};
    neg_e_poses=cg_cache.neg_e_poses_classes{c_idx};

    wl_data_weight=zeros(cg_cache.wl_example_num,1);
    wl_data_weight(pos_e_sel)=wl_data_weight_pos(pos_e_sel);
    wl_data_weight(neg_e_sel)=pair_weights(neg_e_poses);
    
    label_data=cg_cache.wl_labels_classes{c_idx};
    wlearner=train_wl(train_info, label_data, wl_data_weight);
    
    hfeat=apply_wl(train_info, wlearner);
    true_sel=hfeat==label_data;
    
    wl_score=sum(wl_data_weight(true_sel))-sum(wl_data_weight(~true_sel));
           
        
    if wl_score>best_wl_score
        best_wl_score=wl_score;
        best_wl=wlearner;
    end

end

hs=[];
if ~isempty(best_wl)
    hs=best_wl;
end


disp_loop_info(train_info, cache_info, hs);

end




function model=gen_model_mc(model)


hs_num=size(model.hs,1);


w_class_dim_idxes=gen_tensor_prod_sel_idxes(model.class_num, hs_num);
model.w_mc=model.w(w_class_dim_idxes);


end


function model=gen_model_mc_cw(model)

iter_num=size(model.hs,1)/model.class_num;
w_class_dim_idxes=gen_tensor_prod_sel_idxes(model.class_num, iter_num);
model.w_mc=model.w(w_class_dim_idxes);
hs_mc=cell(model.class_num,1);
for c_idx=1:length(hs_mc)
    hs_mc{c_idx}=model.hs(w_class_dim_idxes(c_idx,:),:);
end
model.hs_mc=hs_mc;

end



function tensor_prod_sel_idxes=gen_tensor_prod_sel_idxes(class_num, hs_num, label_idxes)

if nargin<3
    label_idxes=1:class_num;
end

dim_num=hs_num*class_num;
tensor_prod_sel_idxes=zeros(length(label_idxes),hs_num);

for class_idx_idx=1:length(label_idxes)
    class_idx=label_idxes(class_idx_idx);
    tensor_prod_sel_idxes(class_idx_idx,:)=class_idx:class_num:dim_num;
end

end




function pair_feat=calc_pair_feat_mc(train_info, cache_info, wlearners)

     
    
    hfeat=apply_wl(train_info, wlearners);
       
    
    wl_num=size(wlearners,1);
        
    pair_num=cache_info.pair_num;
    class_num=cache_info.class_num;
    tensor_info=cache_info.tensor_info;
    
    pair_feat=zeros(pair_num*class_num,wl_num);
    pair_feat(tensor_info.tensor_left_sel,:)=hfeat(tensor_info.tensor_left_e_idxes,:);
    pair_feat(tensor_info.tensor_right_sel,:)=-hfeat(tensor_info.tensor_right_e_idxes,:);
    
    pair_feat=reshape(pair_feat, pair_num, class_num*wl_num);
   
     
end




function pair_feat=calc_pair_feat_mc_cw(train_info, cache_info, wlearners)

    pair_num=cache_info.pair_num;
    class_num=cache_info.class_num;
    tensor_info=cache_info.tensor_info;
        
    wl_num2=size(wlearners,1);
    iter_num=wl_num2/class_num;
    
    assert(wl_num2==class_num);
    
    
    cached_hfeat=apply_wl(train_info, wlearners);
    
    e_num=size(cached_hfeat,1);
    % assume that wlearners is in an order of 1 to K
    cached_hfeat=reshape(cached_hfeat, e_num*class_num, iter_num);
   
        
    pair_feat=zeros(pair_num*class_num, iter_num);
    pair_feat(tensor_info.tensor_left_sel,:)=cached_hfeat(tensor_info.tensor_left_e_idxes,:);
    pair_feat(tensor_info.tensor_right_sel,:)=-cached_hfeat(tensor_info.tensor_right_e_idxes,:);
    
    pair_feat=reshape(pair_feat, pair_num, class_num*iter_num);
    

    
end










function [struct_data]=do_gen_struct_data_mc(mc_data, use_cw)


label_data=mc_data.label_data;
label_values=mc_data.label_vs;
class_num=length(label_values);
e_num=size(mc_data.feat_data,1);


e_idx_mat=repmat(1:e_num, class_num-1,1);
pair_num=(class_num-1)*e_num;
struct_example_idxes=reshape(e_idx_mat,pair_num,1);
struct_pairs=cat(2, struct_example_idxes, struct_example_idxes);

e_pair_idxes=1:pair_num;
e_pair_idxes=reshape(e_pair_idxes,class_num-1,e_num);
e_pair_idxes=e_pair_idxes';


tensor_y_pairs=zeros(pair_num,2);

tensor_e_poses_left=zeros(pair_num, class_num);
tensor_e_poses_right=zeros(pair_num, class_num);


cached_e_idxes_classes=cell(class_num,1);
neg_e_pair_idxes_classes=cell(class_num,1);


e_idx_offset_classes_cw=(0:class_num-1)*e_num;

pair_counter=0;

for e_idx=1:e_num
    e_label=label_data(e_idx);
    class_sel=label_values==e_label;
    class_idx=find(class_sel,1);
    
    one_tensor_y_pairs=zeros(class_num-1,2);
    one_tensor_y_pairs(:,1)=class_idx;
    one_tensor_y_pairs(:,2)=find(~class_sel);
    
    tensor_y_pairs=cat(1, tensor_y_pairs, one_tensor_y_pairs);
    
       
    
    cached_e_idxes_classes{class_idx}(end+1)=e_idx;

    one_pair_counter=0;
    for c_idx=1:class_num
        if c_idx~=class_idx
            one_pair_counter=one_pair_counter+1;
            pair_idx=pair_counter+one_pair_counter;
            
                        
            if use_cw
                tensor_e_poses_left(pair_idx, class_idx)=e_idx+e_idx_offset_classes_cw(class_idx);
                tensor_e_poses_right(pair_idx, c_idx)=e_idx+e_idx_offset_classes_cw(c_idx);
            else
                tensor_e_poses_left(pair_idx, class_idx)=e_idx;
                tensor_e_poses_right(pair_idx, c_idx)=e_idx;
            end
            
            neg_e_pair_idxes_classes{c_idx}(end+1)=pair_idx;
        end
    end
            
    pair_counter=pair_counter+one_pair_counter;
%     
    
end






% tensor_info.tensor_left_sel_mat=tensor_e_poses_left;
% tensor_info.tensor_right_sel_mat=tensor_e_poses_right;

tensor_e_poses_left=tensor_e_poses_left(:);
tensor_e_poses_right=tensor_e_poses_right(:);
tensor_left_sel=tensor_e_poses_left>0;
tensor_right_sel=tensor_e_poses_right>0;
tensor_left_e_idxes=tensor_e_poses_left(tensor_left_sel);
tensor_right_e_idxes=tensor_e_poses_right(tensor_right_sel);



struct_label_loss=ones(size(struct_pairs,1),1);

struct_cached_feat=mc_data.feat_data;


struct_slack_idxes=struct_example_idxes;

struct_data=[];
struct_data.struct_slack_idxes=struct_slack_idxes;
struct_data.struct_label_loss=struct_label_loss;
struct_data.struct_pairs=struct_pairs;
struct_data.struct_cached_feat=struct_cached_feat;
struct_data.struct_example_idxes=struct_example_idxes;
struct_data.struct_pair_num=length(struct_pairs);
struct_data.struct_e_num=size(struct_cached_feat, 1);

% struct_data.label_vs=label_values;

tensor_info.tensor_left_sel=tensor_left_sel;
tensor_info.tensor_right_sel=tensor_right_sel;
tensor_info.tensor_left_e_idxes=tensor_left_e_idxes;
tensor_info.tensor_right_e_idxes=tensor_right_e_idxes;
tensor_info.tensor_y_pairs=tensor_y_pairs;



struct_cache_info.tensor_info=tensor_info;

% for cg_cache:
struct_cache_info.cached_e_idxes_classes=cached_e_idxes_classes;
struct_cache_info.neg_e_pair_idxes_classes=neg_e_pair_idxes_classes;

struct_cache_info.e_pair_idxes=e_pair_idxes;

struct_data.struct_cache_info=struct_cache_info;


end






function wl_cache=gen_wl_cache(train_info)

wl_cache=[];
feat_data=train_info.struct_data.struct_cached_feat;
wl_cache.feat_data=feat_data;

end



function wlearner=train_wl(train_info, label_data, data_weight)

feat_data=train_info.wl_cache.feat_data;
assert(isa(feat_data, 'uint8'));

pos_sel=label_data>0;
neg_sel=~pos_sel;


tree_data=[];
tree_data.X0=feat_data(neg_sel,:);
tree_data.X1=feat_data(pos_sel,:);

assert( ~isempty(data_weight));
tree_data.wts0=data_weight(neg_sel);
tree_data.wts1=data_weight(pos_sel);


one_model = binaryTreeTrain( tree_data, train_info.pTree );
wlearner={one_model};


end




function hfeat=apply_wl(train_info, wlearners)


feat_data=train_info.wl_cache.feat_data;
assert(isa(feat_data, 'uint8'));

wl_num=length(wlearners);
e_num=size(feat_data,1);

hfeat=zeros(e_num, wl_num);
for wl_idx=1:wl_num

    one_model=wlearners{wl_idx};
    one_hfeat = binaryTreeApply( feat_data, one_model);
    one_hfeat= nonzerosign(one_hfeat);
    hfeat(:, wl_idx)=one_hfeat;

end




end



function disp_loop_info(train_info, cache_info, hs)

fprintf('---multib-cw: class:%d, example:%d, feat:%d, wlearner:%d, solver_mex:%d, stagewise:%d, use_cw:%d\n', ...
    train_info.class_num, train_info.e_num, train_info.feat_num, size(hs,1), ...
    train_info.use_solver_mex, train_info.use_stagewise, train_info.use_cw);

end


