%%Motivation: Plan to predict the maximum time after the actual accident,
%%the time for which its impact is still present

%% 3 learners are present. Each learner learns from a sensor data
%% reward in each case is the percentage of accuracy obtained
%%input context: whether at rush hour, type of accident, speed before
%%accident, (final speed/initial speed) just after the accident, the time for which the impact
%%was felt.
tic;
global ctxt_partition ctxt_dim slice_param learner_num action_dim strategy_num h y explore_set explore_setlabel train_set label_set prediction label

ctxt_dim = 4;
action_dim =55; %% the maximum value present in the dataset
ctxt_max1 = [1 4 75 3];
ctxt_max = ctxt_max1(1:ctxt_dim);		%%% maximum value in each context dimension (adopted dimension)

label_pos = 5;
format = repmat('%f', [1 label_pos]);
z=0.1;
f=2.0;
learner_num=3;
slice_param=2;
strategy_num=4;
T_final=1000; %% number of online instances

%%% each context subspace includes range (bottom left coordinates, upper right coordinates)
ctxt_subspace = struct('left',zeros(1,ctxt_dim),'right',ones(1,ctxt_dim),'reward_learner',zeros(1, learner_num),'action_array_train',zeros(learner_num,learner_num), ...
'action_arr',zeros(learner_num,learner_num),'action_array_ex',zeros(strategy_num,learner_num),'learner_own',zeros(1,learner_num),'reward_strategy',zeros(strategy_num,learner_num),'reward_counter',zeros(strategy_num,learner_num),'level',1);
ctxt_partition = ctxt_subspace; %% set containing the partitions of the space
sub_idx=1;
ParUpdt(sub_idx);





cost_of_calling=zeros(learner_num,learner_num); %% d(i,k) the cost of calling one learner 
fin1 = fopen('Input1.txt');
fin2 = fopen('Input2.txt');
fin3 = fopen('Input3.txt');

ctxt_input = zeros(200, ctxt_dim);		%%% read at most 100 instances
label_input = zeros(200, 1);
size = 0;
input_array_complete=zeros(200,4,learner_num);
label_array_complete=zeros(200,1,learner_num);
c=1;
train_set=zeros(3000,ctxt_dim,learner_num,slice_param^ctxt_dim);
label_set=zeros(3000,1,learner_num,slice_param^ctxt_dim);
explore_set=zeros(3000,ctxt_dim,learner_num,strategy_num,slice_param^ctxt_dim); %% the context sent during the exploration
explore_setlabel=zeros(3000,1,learner_num,strategy_num,slice_param^ctxt_dim);   %% the actual results which could be used for rewards during the exploration.
h=0;
y=0;
x=0;
final=zeros(2500,7);

while 1
    cline = fgetl(fin1);
    if ~ischar(cline)
        break;
    end
    entry = sscanf(cline, format);
    entry = entry';
    size = size + 1;    
    input_array_complete(size,:,1) = entry(1:ctxt_dim)./ctxt_max;   
    label_array_complete(size,1,1) = entry(label_pos);
end
size=0;
while 1
    cline = fgetl(fin2);
    if ~ischar(cline)
        break;
    end
    entry = sscanf(cline, format);
    entry = entry';
    size = size + 1;    
    input_array_complete(size,:,2) = entry(1:ctxt_dim)./ctxt_max;   
    label_array_complete(size,1,2) = entry(label_pos);
end
size=0;
while 1
    cline = fgetl(fin3);
    if ~ischar(cline)
        break;
    end
    entry = sscanf(cline, format);
    entry = entry';
    size = size + 1;    
    input_array_complete(size,:,3) = entry(1:ctxt_dim)./ctxt_max;   
    label_array_complete(size,1,3) = entry(label_pos);
end
store_idx=zeros(1,2000);
train_idx=zeros(1,2000);
exploit_idx=zeros(1,2000);
explore1_idx=zeros(1,2000);
explore2_idx=zeros(1,2000);
a=1;
b=1;
q=1;
qw=1;
qwa=1;

prediction=0;
count_array=zeros(1,4);
for count = 1:learner_num  %% count refers to which learner is being chosen
    size=200;
    for t= 1:20000
        pick = ceil(size*rand);         %%% randomly pick one instance
       
        ctxt = input_array_complete(pick,:,count);  
        label = label_array_complete(pick,1,count);
        [sub_idx] = cmabActSel(ctxt, t);
        %%display(ctxt);
        store_idx(a)=sub_idx;
        a=a+1;
        
        
        
        if(ctxt_partition(sub_idx).action_array_ex(:,count) <=  0.1*t^z*log(t))
            row1=find(ctxt_partition(sub_idx).action_array_ex(:,count) <= 0.1*t^z*log(t));
            k=datasample(row1,1);
            display(k);
               y=y+1;
               count_array(1,1)=count_array(1,1)+1;
               explore_strategy(count,k,ctxt,label,sub_idx,explore_set,explore_setlabel,t); %%count tells which learner wants his arm to be explored
               explore1_idx(q)=sub_idx;
               q=q+1;
              ctxt_partition(sub_idx).action_array_ex(k,count)=ctxt_partition(sub_idx).action_array_ex(k,count)+1;
                ctxt_partition(sub_idx).learner_own(1,count)=ctxt_partition(sub_idx).learner_own(1,count)+1;
        else if(ctxt_partition(sub_idx).action_array_train(:,count)<= 0.2*t^z*log(t))
               row2=find( ctxt_partition(sub_idx).action_array_train(:,count)<=0.2*t^z*log(t));
               l=datasample(setdiff(row2,count),1);
               ctxt_partition(sub_idx).action_array_train(l,count)= ctxt_partition(sub_idx).learner_own(1,l)-ctxt_partition(sub_idx).action_arr(l,count);
               if(ctxt_partition(sub_idx).action_array_train(l,count)<=0.2*t^z*log(t))
                    h=h+1;
                    count_array(1,2)=count_array(1,2)+1;
                    disp('going for training');
                    train_idx(b)=sub_idx;
                    b=b+1;
                    train(l,ctxt,label,sub_idx,train_set,label_set,t);
               end
            else if(ctxt_partition(sub_idx).action_arr(:,count)<= 0.001*t^z*log(t))
                    y=y+1;
                    count_array(1,3)=count_array(1,3)+1;
                    row3=find(ctxt_partition(sub_idx).action_arr(:,count)<= 0.001*t^z*log(t));
                    l=datasample(setdiff(row3,count),1);
                    explore_learner(l,ctxt,label,sub_idx,explore_set,explore_setlabel,t);
                    explore2_idx(qw)=sub_idx;
                    qw=qw+1;
                    ctxt_partition(sub_idx).action_arr(l,count)=ctxt_partition(sub_idx).action_arr(l,count)+1;
                    ctxt_partition(sub_idx).learner_own(1,count)=ctxt_partition(sub_idx).learner_own(1,count)+1;
                    
                else
                    
                    count_array(1,4)=count_array(1,4)+1;
                    [prediction]=exploit(sub_idx,ctxt,label,t);
                    [find_ctxt]=cmabActSel(ctxt, t);
                    exploit_idx(qwa)=sub_idx;
                    qwa=qwa+1;
                    x=x+1;
                    final(x,1)=ctxt(1);
                    final(x,2)=ctxt(2);
                    final(x,3)=ctxt(3);
                    final(x,4)=ctxt(4);
                    final(x,6)=abs(prediction);
                    final(x,5)=label;
                    final(x,7)=find_ctxt;
                          
                    ctxt_partition(sub_idx).learner_own(1,count)=ctxt_partition(sub_idx).learner_own(1,count)+1;

                end
                end
                
        end
    end
    final(all(final==0,2),:)=[];
plot(final(:,6),final(:,5));
%%cmu=confusionmat(final(:,1),final(:,2));
  %%      N = sum(cmu(:));
   %%accuracy = 1-( ( N-sum(diag(cmu)) ) / N);
     
   %%disp('Accuray percentage =');
     %%   display(accuracy*100);
     
     
      accuracy=mean(rdivide(abs(final(:,5)-final(:,6)),final(:,5)));
      accuracy1=sqrt(mean(abs(final(:,6)-final(:,5)).^2));
      disp('percentage error =');
      display(accuracy*100);
      disp('root mean square error=');
      display(accuracy1*1);
       [F,c]=hist(final(:,5),20);
       bar(c,F);
       plot(sqrt(abs(final(:,5)-final(:,6)).^2),final(:,5));
      
toc
end
