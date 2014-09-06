function train(l,ctxt,label,sub_idx,train_set,label_set,t)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
global ctxt_partition ctxt_dim slice_param learner_num action_dim strategy_num h y explore_set explore_setlabel train_set label_set prediction label
%% only form the respective training sets in this phase
train_set(h,:,l,sub_idx)=ctxt;
%% i prepare the training set of a particular learner in a particular context cube
label_set(h,:,l,sub_idx)=label;





end

