function ParUpdt(sub_idx)
global ctxt_partition ctxt_dim slice_param learner_num action_dim strategy_num h y explore_set explore_setlabel train_set label_set prediction label

sub_len = length(ctxt_partition);
subspace = ctxt_partition(sub_idx);
level = subspace.level;

ctxt_partition = [ctxt_partition(1:sub_idx-1) ctxt_partition(sub_idx+1:sub_len)];   %%% delete the current subspace from the partition set

 left = subspace.left;
    right = subspace.right;
    mid = (left + right)/2;
    
     for i = 0 : ((slice_param)^ctxt_dim) - 1		%%% creat (slice_param)^ctxt_dim new subspaces
        midpoint = de2bi(i, ctxt_dim);
        new_left = midpoint.*mid + (1-midpoint).*left;
        new_right = new_left + mid;
        
        new_ctxt_subspace = struct('left',new_left,'right',new_right,'reward_learner',zeros(1, learner_num),'action_array_train',zeros(learner_num,learner_num), ...
'action_arr',zeros(learner_num,learner_num),'action_array_ex',zeros(strategy_num,learner_num),'learner_own',zeros(1,learner_num),'reward_strategy',zeros(strategy_num,learner_num),'reward_counter',zeros(strategy_num,learner_num),'level',level+1);
        ctxt_partition = [new_ctxt_subspace ctxt_partition];		%%% add the new subspace into the partition set
    end

end