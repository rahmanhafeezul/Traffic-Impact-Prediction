function explore_learner(l,ctxt,label,sub_idx,explore_set,explore_setlabel,t)
global ctxt_partition ctxt_dim slice_param learner_num action_dim strategy_num h y explore_set explore_setlabel train_set label_set prediction label
for q= 1:strategy_num
        explore_set(y,:,l,q,sub_idx)=ctxt;
        explore_setlabel(y,:,l,q,sub_idx)=label;


end


%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


end

