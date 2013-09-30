

function train_info=boost_learn_config(train_info)


struct_data=train_info.struct_data;

if ~isfield(train_info, 'wlearner_weight_init_mode')
    train_info.wlearner_weight_init_mode=2;
end


slack_group_info=gen_slack_group_info(struct_data.struct_slack_idxes);
train_info.slack_group_poses=slack_group_info.slack_group_poses;
train_info.slack_group_idxes=slack_group_info.slack_group_idxes;


% ==================== extend by other application, eg: multiclass

train_info.find_wlearner_fn=[];
train_info.calc_pair_feat_fn=[];
train_info.gen_cache_info_fn=[];


% ===================== callback function setting: 
        
    train_info.init_work_info_fn=@init_work_info;
    train_info.update_wlearner_info_fn=@update_wlearner_info;

%==========================

end




function work_info=update_wlearner_info(train_info, work_info, wlearner_update_info)


if work_info.solver_init_iter
    
    
    label_losses=train_info.struct_data.struct_label_loss;
    work_info.pair_label_losses=label_losses;
    work_info.pair_idxes=(1:length(label_losses))';
    work_info.sel_pair_num=length(label_losses);
    work_info.pair_sub_idx_groups=train_info.slack_group_poses;
    work_info.slack_group_idxes=train_info.slack_group_idxes;
    work_info.slack_num=work_info.slack_num;
    work_info.pair_feat=[];
       
    
       
    
end


work_info.new_wlearner=[];

work_info=do_update_wlearner_info(train_info, work_info, wlearner_update_info);
new_pair_feat=train_info.calc_pair_feat_fn(train_info, work_info.cache_info, work_info.new_wlearner);


%work_info.pair_feat=cat(2,work_info.pair_feat, new_pair_feat);
work_info.new_pair_feat=new_pair_feat;


end



function work_info=do_update_wlearner_info(train_info, work_info, wlearner_update_info)


cache_info=work_info.cache_info;


if work_info.solver_init_iter
       
    wlearner_weight_init_mode=train_info.wlearner_weight_init_mode;
    
    % =============================== 1. select pairs for init accding to slack group 
    
    if wlearner_weight_init_mode==1
            
        pair_idxes=work_info.pair_idxes;
        pair_sub_idx_groups=work_info.pair_sub_idx_groups;
        tmp_idxes=randsample(length(pair_idxes), length(pair_sub_idx_groups));
        pair_weights=zeros(length(pair_idxes), 1);
        pair_weights(tmp_idxes)=1;
        
    end
    
    % ===============================

    
    % =============================== 2. use all pair to init
    
    if wlearner_weight_init_mode==2
        pair_num=length(train_info.struct_data.struct_label_loss);
        pair_weights=ones(pair_num,1);
    end

    % ===============================

    wlearner_update_info.pair_weights=pair_weights;
end


new_wlearner=train_info.find_wlearner_fn(train_info, cache_info, wlearner_update_info.pair_weights);


work_info.new_wlearner=new_wlearner;
work_info.wlearners=cat(1, work_info.wlearners, new_wlearner);
work_info.wlearner_num=size(work_info.wlearners,1);


end









function [work_info train_info]=init_work_info(train_info)


work_info.wlearners=[];
%it's required for cp solver
work_info.slack_num=length(train_info.slack_group_poses);
work_info.pair_feat=[];
work_info.pair_num_total=length(train_info.struct_data.struct_label_loss);

[work_info.cache_info train_info]=train_info.gen_cache_info_fn(train_info);

end




function slack_group_info=gen_slack_group_info(slack_idxes)


pair_num=length(slack_idxes);
slack_idxes_vs=unique(slack_idxes);
slack_num=length(slack_idxes_vs);


slack_group_idxes=zeros(pair_num,1);

if slack_num==pair_num
    c_group_poses=(1:pair_num)';
    c_group_type=3;
    slack_group_idxes(:,1)=1:pair_num;
else
    
    c_group_type=1;
    
end

if c_group_type==1
    c_group_poses=cell(slack_num,1);
    for t_idx=1:length(slack_idxes_vs)
        slack_idx_v=slack_idxes_vs(t_idx);
        pos_sel=(slack_idxes==slack_idx_v);
        poses=find(pos_sel>0);
        % make the poses as a row vector, consist with c_group_type 2
        c_group_poses{t_idx}=poses';
        slack_group_idxes(pos_sel)=t_idx;
    end
end

c_group_poses_cell=c_group_poses;
if ~iscell(c_group_poses_cell)
    c_group_poses_cell=num2cell(c_group_poses_cell,2);
end

slack_group_info.slack_group_poses_mat=c_group_poses;
slack_group_info.slack_group_poses=c_group_poses_cell;
slack_group_info.slack_group_type=c_group_type;
slack_group_info.slack_group_num=slack_num;
slack_group_info.slack_group_idxes=slack_group_idxes;

end


