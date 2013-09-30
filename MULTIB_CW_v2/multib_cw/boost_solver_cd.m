
function [solver_result work_info]=boost_solver_cd(train_info, work_info)


use_stagewise=train_info.use_stagewise;
if use_stagewise
    [solver_result work_info]=call_solver_stage(train_info, work_info);
else
    [solver_result work_info]=call_solver_fast(train_info, work_info);
end




end






function [solver_result work_info]=call_solver_fast(train_info, work_info)


max_ws_iter=2;
if isfield(train_info, 'max_ws_iter')
    max_ws_iter=train_info.max_ws_iter;
end

tradeoff_nv=1e-5;
if isfield(train_info, 'max_ws_iter')
    tradeoff_nv=train_info.tradeoff_nv;
end

use_stagewise=train_info.use_stagewise;
if use_stagewise
    tradeoff_nv=0;
end


use_mex=train_info.use_solver_mex;
% use_mex=false;


new_wl_num=size(work_info.new_pair_feat, 2);

train_cache=[];
if isfield(work_info, 'train_cache_sub')
    train_cache=work_info.train_cache_sub;
end

if isempty(train_cache)
      
    
    label_losses=work_info.pair_label_losses;
        
    if use_mex
        train_cache.p_sel_mat=int8([]);
        train_cache.n_sel_mat=int8([]);
    else
        train_cache.p_sel_mat=false(0);
        train_cache.n_sel_mat=false(0);
    end
        
    train_cache.exp_m_wh=label_losses;
    old_wl_num=0;

else
    
    old_wl_num=length(train_cache.w);
end

wl_num=old_wl_num+new_wl_num;    
work_info.solver_feat_change_idxes=(old_wl_num+1:wl_num)';
train_cache.w(work_info.solver_feat_change_idxes)=0;

work_info.solver_wl_valid_sel=true(wl_num, 1);
work_info.solver_update_wl_idxes=work_info.solver_feat_change_idxes;


train_cache.use_mex=use_mex;


train_cache=update_cache_feat_change(work_info, train_cache);
update_dim_idxes=work_info.solver_update_wl_idxes;
assert(~isempty(update_dim_idxes))
    
if ~use_stagewise
    if max_ws_iter>1
        valid_dim_idxes=find(work_info.solver_wl_valid_sel);
        for ws_idx=2:max_ws_iter
            valid_dim_idxes=valid_dim_idxes(randperm(length(valid_dim_idxes)));
            update_dim_idxes=cat(1, update_dim_idxes, valid_dim_idxes);
        end
    end
end



init_w=train_cache.w;
exp_m_wh=train_cache.exp_m_wh;
    

    if use_mex
        
                
        [w exp_m_wh]=boost_solver_mex_simple(tradeoff_nv, update_dim_idxes, init_w,...
            train_cache.p_sel_mat, train_cache.n_sel_mat, exp_m_wh);
        failed=false;
        
        method_name='cd_mex';
        
    else
 
        if use_stagewise
                       
            [w exp_m_wh failed]=do_solve_stage_debug(tradeoff_nv, update_dim_idxes, init_w,...
                train_cache.p_sel_mat, train_cache.n_sel_mat, exp_m_wh);
            method_name='cd_stage_debug';
            
        else
            
            [w exp_m_wh failed]=do_solve(tradeoff_nv, update_dim_idxes, init_w,...
                train_cache.p_sel_mat, train_cache.n_sel_mat, exp_m_wh);
            method_name='cd';
                       
        end
    end
    
    
    obj_value=sum(w)*tradeoff_nv + sum(exp_m_wh);
    wl_data_weight=exp_m_wh;
    
    



iter_num=length(update_dim_idxes);


train_cache.w=w;
train_cache.exp_m_wh=exp_m_wh;
work_info.train_cache_sub=train_cache;


solver_result.method=method_name;
solver_result.w=w;
solver_result.iter_num=iter_num;
solver_result.obj_value=obj_value;
solver_result.wlearner_pair_weight=wl_data_weight;
solver_result.failed=failed;


fprintf('---solver_info: new_wl_num:%d(%d), tradeoff_nv:%.4f, max_ws_iter:%d, use_mex:%d, use_stagewise:%d\n',...
    new_wl_num, wl_num, tradeoff_nv, max_ws_iter, use_mex, use_stagewise);


end




function [w exp_m_wh failed]=do_solve(tradeoff_nv, update_dim_idxes, init_w, p_sel_mat, n_sel_mat, exp_m_wh)

    one_feat_v=1;

    tmp_v1=tradeoff_nv/2;
    tmp_v2=tmp_v1^2;


    w=init_w;
    pair_num=size(p_sel_mat, 1);
            
    for up_idx=1:length(update_dim_idxes)
        
        up_dim_idx=update_dim_idxes(up_idx);
        p_sel=p_sel_mat(:, up_dim_idx);
        n_sel=n_sel_mat(:, up_dim_idx);

        one_old_w=w(up_dim_idx);
        if one_old_w>0
            
           tmp_m_wh=zeros(pair_num, 1); 
           tmp_m_wh(n_sel)=-one_old_w*one_feat_v;
           tmp_m_wh(p_sel)=one_old_w*one_feat_v;
           exp_m_wh=exp_m_wh.*exp(tmp_m_wh);
        end

        V_p=sum(exp_m_wh(p_sel));
        
        one_new_w=0;
        error_num=nnz(n_sel);
        if error_num>0
            V_m=sum(exp_m_wh(n_sel));
            if V_p>V_m && V_m>eps
               one_new_w=log(sqrt(V_p*V_m+tmp_v2) - tmp_v1) - log(V_m);
            end
        end
        

        w(up_dim_idx)=one_new_w;

        tmp_m_wh=zeros(pair_num, 1); 
        tmp_m_wh(p_sel)=-one_new_w*one_feat_v;
        tmp_m_wh(n_sel)=one_new_w*one_feat_v;
        exp_m_wh=exp_m_wh.*exp(tmp_m_wh);

    end
    
    failed=false;
    if one_new_w<=0
        failed=true;
    end
    
    if failed
        
        fprintf('\n-----solver failed: error_num:%d, V_p:%.2e, V_m:%.2e\n\n', error_num, V_p, V_m);
        
    end

end




function [w exp_m_wh failed]=do_solve_stage_debug(tradeoff_nv, update_dim_idxes, init_w, p_sel_mat, n_sel_mat, exp_m_wh)

    w=init_w;
    pair_num=size(p_sel_mat, 1);
    
    assert(tradeoff_nv==0);
    
    one_feat_v=1;
    
                
    for up_idx=1:length(update_dim_idxes)
        
        up_dim_idx=update_dim_idxes(up_idx);
        
        p_sel=p_sel_mat(:, up_dim_idx);
        n_sel=n_sel_mat(:, up_dim_idx);

        V_p=sum(exp_m_wh(p_sel));


        one_old_w=w(up_dim_idx);
        assert( one_old_w==0);

        one_new_w=0;
        error_num=nnz(n_sel);
        if error_num>0
            V_m=sum(exp_m_wh(n_sel));
            if V_p>V_m && V_m>eps
                one_new_w=0.5*log(V_p/V_m);
            end
        end


        w(up_dim_idx)=one_new_w;

        tmp_m_wh=zeros(pair_num, 1); 
        tmp_m_wh(p_sel)=-one_new_w*one_feat_v;
        tmp_m_wh(n_sel)=one_new_w*one_feat_v;
        exp_m_wh=exp_m_wh.*exp(tmp_m_wh);


        failed=false;
        if one_new_w<=0
            failed=true;
        end


        if failed

            fprintf('\n------solver failed: error_num:%d, V_p:%.2e, V_m:%.2e\n\n', error_num, V_p, V_m);

        end
    
    end
    

end






function train_cache=update_cache_feat_change(work_info, train_cache)

use_mex=train_cache.use_mex;
new_w_dim_idxes=work_info.solver_feat_change_idxes;

if ~isempty(new_w_dim_idxes)
    train_cache=update_cache_remove_dim(train_cache, new_w_dim_idxes);

    new_p_sel_mat=work_info.new_pair_feat>0.5;


    % for multiclass case, pair feat may have the value 0, so don't do this
%     new_n_sel_mat=~new_p_sel_mat;

    new_n_sel_mat=work_info.new_pair_feat<-0.5;


    assert(size(new_p_sel_mat, 2)==length(new_w_dim_idxes));

    
    if use_mex
        train_cache.p_sel_mat(:, new_w_dim_idxes)=int8(new_p_sel_mat);
        train_cache.n_sel_mat(:, new_w_dim_idxes)=int8(new_n_sel_mat);
    else
        train_cache.p_sel_mat(:, new_w_dim_idxes)=new_p_sel_mat;
        train_cache.n_sel_mat(:, new_w_dim_idxes)=new_n_sel_mat;
    end
end

end




function train_cache=update_cache_remove_dim(train_cache, new_w_dim_idxes)

exp_m_wh=train_cache.exp_m_wh;
init_w=train_cache.w;
for tmp_idx=1:length(new_w_dim_idxes)
    one_w_idx=new_w_dim_idxes(tmp_idx);
    one_old_w=init_w(one_w_idx);
    if one_old_w>0
       n_sel=logical(train_cache.n_sel_mat(:, one_w_idx));
       pair_num=length(n_sel);
       tmp_m_wh=repmat(one_old_w, pair_num, 1); 
       tmp_m_wh(n_sel)=-tmp_m_wh(n_sel);
       exp_m_wh=exp_m_wh.*exp(tmp_m_wh); 
       
       init_w(one_w_idx)=0;
    end
end

train_cache.w=init_w;
train_cache.exp_m_wh=exp_m_wh;


end

















function [train_result_sub work_info]=call_solver_stage(train_info, work_info)


% using exp loss

% only

label_losses=work_info.pair_label_losses;
new_pair_feat=work_info.new_pair_feat;
new_w_dim=size(new_pair_feat,2);


last_train_cache=[];
if isfield(work_info, 'train_cache_sub')
    last_train_cache=work_info.train_cache_sub;
end


if ~isempty(last_train_cache)
    last_w_dim=length(last_train_cache.w);
    init_w=last_train_cache.w;
    exp_m_wh=last_train_cache.exp_m_wh;
else
    last_w_dim=0;
    init_w=[];
    exp_m_wh=label_losses;
end


w_dim=last_w_dim+new_w_dim;
ws_dim_idxes=last_w_dim+1:w_dim;
init_w(ws_dim_idxes)=0;


w=init_w;

for nw_idx=1:length(ws_dim_idxes)
    
    nw_dim=ws_dim_idxes(nw_idx);
    f_j=new_pair_feat(:, nw_idx);
    p_sel=f_j>0.5;
    n_sel=f_j<-0.5;

    one_old_w=w(nw_dim);
    assert(one_old_w==0);
       
    
    V_p=sum(exp_m_wh(p_sel));

    one_new_w=0;
    error_num=nnz(n_sel);
    if error_num>0
        V_m=sum(exp_m_wh(n_sel));
        if V_p>V_m && V_m>eps
            one_new_w=0.5*log(V_p/V_m);
        end
    end
    
        
    one_new_w=max(one_new_w,0);
    w(nw_dim)=one_new_w;

    exp_m_wh=exp_m_wh.*exp(-f_j.*one_new_w);
    
    failed=false;
    if one_new_w<=0
        failed=true;
    end
    
    if failed

%         fprintf('\n------solver failed: error_num:%d, V_p:%.2e, V_m:%.2e\n\n', error_num, V_p, V_m);

    end

end



cur_mu=exp_m_wh;
obj_value=sum(cur_mu);


train_cache=[];
train_cache.w=w;
train_cache.exp_m_wh=exp_m_wh;
work_info.train_cache_sub=train_cache;

train_result_sub.method='stage';
train_result_sub.w=w;
train_result_sub.obj_value=obj_value;

train_result_sub.wlearner_pair_weight=cur_mu;
end


