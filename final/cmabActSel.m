function [sub_idx] = cmabActSel(ctxt, t)
%%% action selection
global ctxt_partition ctxt_dim slice_param learner_num action_dim strategy_num h y explore_set explore_setlabel train_set label_set prediction label

    %%%% determine subspace
    sub_len = length(ctxt_partition);
    for i = 1 : sub_len
        subspace = ctxt_partition(i);
        ifLeft = min(ctxt >= subspace.left);
        ifRight = min(ctxt <= subspace.right);
        if ifLeft && ifRight
            sub_idx = i;
            break;
        end
    end
    
    
end
