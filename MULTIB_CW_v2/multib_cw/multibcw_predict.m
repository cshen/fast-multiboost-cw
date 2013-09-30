
function predict_result=multibcw_predict(model, test_data, predict_config)

if nargin<3
    predict_config=[];
end

  

feat_data=test_data.feat_data;



if isfield(model, 'use_cw') && model.use_cw
    
    hs_feat=cell(model.class_num, 1);
    
    hs_mc=model.hs_mc;
    for c_idx=1:model.class_num
        one_hs=hs_mc{c_idx};
        one_hs_feat=apply_wl(feat_data, one_hs);
        hs_feat{c_idx}=one_hs_feat;
    end
    iter_num=size(hs_mc{1},1);
else
    
    hs_feat=apply_wl(feat_data, model.hs);
    iter_num=size(model.hs,1);
end

[error_rate predict_labels]= do_predict_one(test_data, model, hs_feat, iter_num);

predict_result.error_rate=error_rate;
predict_result.predict_labels=predict_labels;
predict_result.iter_num=iter_num;


if ~isempty(predict_config)
    eva_iter_idxes=predict_config.eva_iter_idxes;
        
    error_rate_iters=zeros(1,length(eva_iter_idxes));
    for iter_idx_idx=1:length(eva_iter_idxes)
        
        iter_idx=eva_iter_idxes(iter_idx_idx);
        
        fprintf('---boost_predict, iter_idx:%d/%d\n', iter_idx, eva_iter_idxes(end));
        
        error_rate = do_predict_one(test_data, model, hs_feat, iter_idx);
        error_rate_iters(iter_idx_idx)=error_rate;
                
    end
    predict_result.error_rate_iters=error_rate_iters;
    predict_result.eva_iter_idxes=eva_iter_idxes;
end

fprintf('\n\n');

end


function [error_rate predict_labels]= do_predict_one(test_data, model, hs_feat, iter_idx)




use_cw=false;
if isfield(model, 'use_cw') && model.use_cw
    use_cw=true;
end

    
    iter_idx=min(iter_idx, size(model.w_mc,2));
    w_mc=model.w_mc(:,1:iter_idx);


if use_cw
    hs_feat2=cell(model.class_num,1);
    for c_idx=1:model.class_num
        hs_feat2{c_idx}=hs_feat{c_idx}(:,1:iter_idx);
    end
    hs_feat=hs_feat2;
else
    
        hs_feat=hs_feat(:,1:iter_idx);
    
end

if use_cw
    scores=[];
    for c_idx=1:model.class_num
        one_scores=hs_feat{c_idx}*w_mc(c_idx,:)';
        scores=cat(2, scores, one_scores);
    end
else
    % hs_feat(hs_feat==0)=-1;
    scores=hs_feat*w_mc';
end

[~, predict_label_idxes]=max(scores,[], 2);
   
label_values=model.label_vs;   
predict_labels=label_values(predict_label_idxes);
error_rate=nnz(predict_labels~=test_data.label_data)/length(scores);

end







function hfeat=apply_wl(feat_data, wlearners)

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


