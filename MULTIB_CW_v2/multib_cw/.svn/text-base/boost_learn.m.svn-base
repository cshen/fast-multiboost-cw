

function train_result=boost_learn(train_info)




fprintf('\n\n############### training start, train_id:%s #################\n\n',train_info.train_id);



if ~isfield(train_info,'max_run_time')
    train_info.max_run_time=inf; %inf hours
end

if ~isfield(train_info,'solver_type')
    train_info.solver_type='custom';
end

if ~isfield(train_info,'max_iteration_num')
    train_info.max_iteration_num=1000;
end

if ~isfield(train_info,'solver_fn')
    train_info.solver_fn=[];
end

if ~isfield(train_info,'notes');
    train_info.notes='';
end


if ~isfield(train_info,'update_train_result_fn')
    train_info.update_train_result_fn=[];
end


[train_info_ext.work_info train_info]=train_info.init_work_info_fn(train_info);
[train_result train_result_ext]=do_train(train_info, train_info_ext);
    

update_train_result_fn=train_info.update_train_result_fn;
if ~isempty(update_train_result_fn)
    train_result=update_train_result_fn(train_result, train_result_ext.work_info, train_info);
end



fprintf('\n\n############### training finished, train_id:%s #################\n\n',train_info.train_id);


end




%========================================================================================================================================================


function [train_result train_result_ext]=do_train(train_info, train_info_ext)

train_id=train_info.train_id;
max_iteration_num=train_info.max_iteration_num;


train_cache.wlearners=[];
train_cache.wl_model=[];
train_cache.obj_value_iters=[];
train_cache.wlearner_time_iters=[];
train_cache.method_time_iters=[];
train_cache.method_time_nocg_iters=[];
train_cache.wlearner_iter_idx=[];


wlearner_iter_idx=0;
cached_method_time=0;


train_finished=false;
method_tstart=tic;
method_time=0;


wlearner_update_info=[];

work_info=train_info_ext.work_info;

work_info.solver_init_iter=true;


while ~train_finished

%-----------------------------check converge------------------------------

    if wlearner_iter_idx>=max_iteration_num 
        fprintf(' \n ---------------------- reach max iteration, %d >= %d \n',wlearner_iter_idx,max_iteration_num);
        train_finished=true;
    end
     
            
    if train_finished
        break;
    end
    
%-----------------------------check converge end ------------------------------


    wlearner_iter_idx=wlearner_iter_idx+1;
    wlearner_update_info.wlearner_iter_idx=wlearner_iter_idx;        
    
    one_wlearner_ts=tic;
    work_info=train_info.update_wlearner_info_fn(train_info, work_info, wlearner_update_info);
    wlearner_time_iter=toc(one_wlearner_ts);
    
    ts_one_cg_train=tic;
    [train_result_sub work_info]=do_train_sub(train_info, work_info);
    opt_time_iter=toc(ts_one_cg_train);
               
        
    
    % required two fields in the call back
    wlearner_num=work_info.wlearner_num;
    pair_num=work_info.sel_pair_num;
    
    
    if isnan(train_result_sub.obj_value)
%         keyboard
        error('objective is NAN');
    end
   
        
    cur_w=full(train_result_sub.w);
    wlearner_update_info.pair_weights=train_result_sub.wlearner_pair_weight;
    if isfield(train_result_sub, 'wlearner_pair_idxes')
        wlearner_update_info.pair_idxes=train_result_sub.wlearner_pair_idxes;
    end
    wlearner_update_info.w=cur_w;
            
    
      
    
    cur_obj_value=train_result_sub.obj_value;
                  
    
    fprintf(['---boost_learn: trn_id:%s, wl_iter:%d/%d, wl_num:%d\n'],...
        train_id, wlearner_iter_idx, max_iteration_num, wlearner_num);           
    fprintf(['---boost_learn: method_time:%.1f, wlearner_t:%.1f, opt_t:%.1f, solver:%s, obj:%.6f \n\n'],...
        method_time, wlearner_time_iter, opt_time_iter, train_info.solver_type, cur_obj_value);
    

    train_cache.w=cur_w;
    train_cache.w_iters{wlearner_iter_idx}=cur_w;
    train_cache.wlearners=work_info.wlearners;
    train_cache.cur_obj=cur_obj_value;
    train_cache.obj_value_iters(wlearner_iter_idx)=cur_obj_value;
    
    train_cache.wlearner_iter_idx=wlearner_iter_idx;
        
    train_cache.wlearner_time_iters(wlearner_iter_idx)=wlearner_time_iter;
    train_cache.opt_time_iters(wlearner_iter_idx)=opt_time_iter;
    method_time=toc(method_tstart)+cached_method_time;
    train_cache.method_time_iters(wlearner_iter_idx)=method_time;
    
    train_cache.pair_num=pair_num;
    train_cache.wlearner_num=wlearner_num;
    
        
    if isfield(work_info, 'wl_model')
        train_cache.wl_model=work_info.wl_model;
    end
    
        
    work_info.solver_init_iter=false;
	    
end


train_result=gen_train_result(train_info, train_cache);
train_result_ext.work_info=work_info;

end



function model=gen_model(train_info, train_cache)

if ~isfield(train_info,'ext_model_info')
    train_info.ext_model_info=[];
end

model.notes=train_info.notes;
model.name=train_info.notes;
model.ext_model_info=train_info.ext_model_info;
model.train_id=train_info.train_id;

model.w=train_cache.w;
model.hs=train_cache.wlearners;
model.wl_model=train_cache.wl_model;

end




function train_result=gen_train_result(train_info, train_cache)

train_result=train_cache;

train_result.model=gen_model(train_info, train_cache);

wlearner_iter_idx=train_cache.wlearner_iter_idx;
train_result.wlearner_iter_num=wlearner_iter_idx;
train_result.obj_value=train_cache.cur_obj;
train_result.method_time=train_cache.method_time_iters(end);
train_result.wlearner_time=sum(train_cache.wlearner_time_iters);
train_result.opt_time=sum(train_cache.opt_time_iters);
train_result.train_id=train_info.train_id;

end







function [train_result_sub work_info]=do_train_sub(train_info, work_info)
        
    [train_result_sub work_info]=train_info.solver_fn(train_info, work_info);

end





