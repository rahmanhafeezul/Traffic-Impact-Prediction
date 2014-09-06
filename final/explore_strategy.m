function explore_strategy(count,k,ctxt,label,sub_idx,explore_set,explore_setlabel,t)
%% in this phase,form the exploration sets which would enable to determine the accuracy of the trained classifiers.
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
global ctxt_partition ctxt_dim slice_param learner_num action_dim strategy_num h y explore_set explore_setlabel train_set label_set prediction label

explore_set(y,:,count,k,sub_idx)=ctxt;
explore_setlabel(y,:,count,k,sub_idx)=label;
end

