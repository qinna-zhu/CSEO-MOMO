function [celue_save,hx,hf,NFE,MaxNFE,Archive_FEs,Archive_convergence,Action_set] = CSEO_MOMO(FUN,Dim,L_bound,U_bound)
% Parameters setting
NFE = 0;
MaxNFE = 1000;                    
Archive_FEs = zeros(MaxNFE,2);  
Archive_convergence = zeros(1,MaxNFE);   
flag_num=0; %Number of stagnation
repeat_num=0; 
% Initial LHS samples
if Dim <100
    initial_sample_size = 50;  
    NP=50;
elseif Dim >= 100
    initial_sample_size = 100;   
    NP=100;
end
%Initialize the database by Latin Hypercube Sampling
sam = repmat(L_bound,initial_sample_size,1)+(repmat(U_bound,initial_sample_size,1)-repmat(L_bound,initial_sample_size,1)).*lhsdesign(initial_sample_size,Dim);
fit = zeros(1,initial_sample_size);
for i=1:initial_sample_size
    fit(i) = FUN(sam(i,:));
    NFE = NFE+1;
    Archive_FEs(NFE,:) = [NFE,fit(i)];
    Archive_convergence(1,i) = min(Archive_FEs(1:NFE,2));
end
dlta = min(sqrt(0.000001^2*Dim),0.00005*sqrt(Dim)*min(U_bound-L_bound));
celue_save=[];
% Build database  
hx = sam; hf = fit;
[~,sidx] = sort(hf);
hx = hx(sidx,:);  hf = fit(sidx);  % history data

%Initialize iterative population 
P=hx;  fitness=hf;   

% DE parameters 
F = 0.5; CR = 0.8;

% Cauchy distribution scale parameter
scale_param_F = 0.1; scale_param_CR = 0.1; 

% RL parameters and initialization  
alp = 0.1;
gamma = 0.9;
Q_Agent = [ 
    0.25 0.25 0.25 0.25;
    0.25 0.25 0.25 0.25;
    0.25 0.25 0.25 0.25;
    0.25 0.25 0.25 0.25;
    0.25 0.25 0.25 0.25;
    0.25 0.25 0.25 0.25;
    0.25 0.25 0.25 0.25;
    0.25 0.25 0.25 0.25;
    ];
State = 1;

% number of calling sampling Actions  
Action1 = 0;
Action2 = 0;
Action3 = 0;
Action4 = 0;
Action_set=[];


% Main loop
while NFE <= MaxNFE
    % Update state and action
    R = 0; action_success = 0;
    Qvalue1 = Q_Agent(State,:);
    temp = exp(Qvalue1);
    ratio = cumsum(temp)/sum(temp);
    jtemp = find(rand(1)<ratio);
    Action = jtemp(1);
      
    % Agent log
    log_Q_Agent{NFE} = Q_Agent;
    log_State{NFE} = State;
    log_ratio{NFE} = ratio;
    log_Action{NFE} = Action;
    disp(['NFE: ' num2str(NFE) ' Action:' num2str(Action)] );
    
    [~,sidx] = sort(hf);
    hx = hx(sidx,:);
    hf = hf(sidx);  %Sort history data in ascending order of fitness  
    
    P=hx(1:NP,:);    %first NP top-ranking samples to comprise the iterative population
    fitness=hf(1:NP);
    
    %winner population and loser population are generated (by pair competition)
    rlist = randperm(NP); % generate random pairs
    rpairs = [rlist(1:ceil(NP/2)); rlist(floor(NP/2) + 1:NP)]';
    
    center = ones(ceil(NP/2),1)*mean(P); % calculate the center position
    
    % do pairwise competitions
    mask = (fitness(rpairs(:,1)) > fitness(rpairs(:,2)));
    mask=mask';
    losers = mask.*rpairs(:,1) + ~mask.*rpairs(:,2);   %Loser sub-population individual index
    winners = ~mask.*rpairs(:,1) + mask.*rpairs(:,2);  %Winner sub-population individual index
   
   
    a1 = 100;
    Max_NFE = a1*Dim+1000; 
    minerror = 1e-20;  
    NS=2*(Dim+1);
    %Tarin data for surrogate
    phdis=real(sqrt(P.^2*ones(size(hx'))+ones(size(P))*(hx').^2-2*P*(hx')));
    [~,sidx]=sort(phdis,2);
    nidx=sidx; nidx(:,NS+1:end)=[];
    nid=unique(nidx);
    train_hx=hx(nid,:);   train_hf=hf(nid);
    DS_train=length(train_hf);

    LB = repmat((L_bound),NP,1);
    UB = repmat((U_bound),NP,1);

    
   Action_set=[Action_set,Action];
    %% Execute action and obtain new data%%
    if Action == 1      
        % Action 1: choose RBF1
        Action1 = Action1 + 1;
        % RBF1 parameters
        ghxd=real(sqrt(train_hx.^2*ones(size(train_hx'))+ones(size(train_hx))*(train_hx').^2-2*train_hx*(train_hx')));
        spr=max(max(ghxd))/(Dim*DS_train)^(1/Dim);
        % build RBF1 network
        i =Action;
        h = 4*(i-1);
        spr= spr * 2^h;
        net=newrbe(train_hx',train_hf,spr);
        Model_FUN=@(x) sim(net,x');
        
    elseif  Action == 2
        % Action 2:  choose RBF2
        Action2 = Action2 + 1;
        % RBF2 parameters
        ghxd=real(sqrt(train_hx.^2*ones(size(train_hx'))+ones(size(train_hx))*(train_hx').^2-2*train_hx*(train_hx')));
        spr=max(max(ghxd))/(Dim*DS_train)^(1/Dim);
        % build RBF2 network
        i =Action;
        h = 4*(i-1);
        spr= spr * 2^h;
        net=newrbe(train_hx',train_hf,spr);
        Model_FUN=@(x) sim(net,x');
        
    elseif  Action == 3
        % Action 3: choose RBF3
        Action3 = Action3 + 1;
        % RBF3 parameters
        ghxd=real(sqrt(train_hx.^2*ones(size(train_hx'))+ones(size(train_hx))*(train_hx').^2-2*train_hx*(train_hx')));
        spr=max(max(ghxd))/(Dim*DS_train)^(1/Dim);
        %  build RBF3 network
        i =Action;
        h = 4*(i-1);
        spr= spr * 2^h;
        net=newrbe(train_hx',train_hf,spr);
        Model_FUN=@(x) sim(net,x');
        
    elseif  Action == 4
        % Action 4: choose RBF4
        Action4 = Action4 + 1;
        % RBF4 parameters
        ghxd=real(sqrt(train_hx.^2*ones(size(train_hx'))+ones(size(train_hx))*(train_hx').^2-2*train_hx*(train_hx')));
        spr=max(max(ghxd))/(Dim*DS_train)^(1/Dim);
        % build RBF4 network
        i =Action;
        h = 4*(i-1);
        spr= spr * 2^h;
        net=newrbe(train_hx',train_hf,spr);
        Model_FUN=@(x) sim(net,x');
    end
    
    %% Evolve the Winner sub-population to generate a trial population
    P1=P(winners,:);
    fitness1=fitness(winners);

    candidate_position=[];
    candidate_position_size=0;

    a1 = 100;Max_NFE = a1*Dim+1000; minerror = 1e-20;   
    % obtain model_best
    [muta_best,~] = JADE(Dim, Max_NFE,Model_FUN, minerror,P,1);


   %Cross-Learning of Gbest and GbestRBF
    temp=hx(1,:);   
    for ii=1:Dim
        temp(ii)=muta_best(ii);
        if Model_FUN(temp)>hf(1)
            temp(ii)=hx(1,ii);
        end        
    end
    
        overallMaxValue = max(P1(:));  
        overallMinValue = min(P1(:));  
        U1=[];
        for ii=1:3
            P1=P(winners,:);
            F_new= F  + cauchyMutation(scale_param_F);
            CR_new = CR + cauchyMutation(scale_param_CR);
            F_new= max(0.4, min(2, F_new));
            CR_new= max(0, min(1, CR_new));
           if  ii==1 
                based_vector=hx(1,:);
           elseif ii==2 
                based_vector=muta_best;
           else
               based_vector=temp;    
               numIndividualsToReinit=ceil(((MaxNFE-NFE)/MaxNFE)*(NP/2)); 
                for iii = 1:numIndividualsToReinit
                    overallMinValue_array = ones(1, Dim) * overallMinValue;  
                    overallMaxValue_array = ones(1, Dim) * overallMaxValue;
                    P1(iii, :) = randomInitialization(overallMinValue_array,overallMaxValue_array); 
                end       
           end
            U = DEoperating(P1,NP/2,Dim,based_vector,F_new,CR_new,UB,LB);  
            U1=[U1;U];
        end
    fitnessModel=Model_FUN(U1);
    [~,sidx] = sort(fitnessModel);
    
    rand_num= size(U1,1);
    pop_rank=U1(sidx(1:rand_num),:);
    if rand_num>1
        mean_pop=mean(pop_rank);
    else
        mean_pop=pop_rank;
    end

    bestU1=U1(sidx(1),:);
    fitnessModelU1=fitnessModel(sidx(1));
     
    dx1=min(sqrt(sum((repmat(U1(sidx(1),:),size(hx,1),1)-hx).^2,2)));
    dx5=min(sqrt(sum((repmat(mean_pop,size(hx,1),1)-hx).^2,2)));
    
        if dx5>dlta && Model_FUN(mean_pop)<fitnessModel(1)
            candidate_position=[candidate_position;mean_pop];
        end
        if  fitnessModel(sidx(1))<hf(1)  && dx1>dlta
            candidate_position=[candidate_position;U1(sidx(1),:)];
        end
        
    
    %real evaluate
    if isempty(candidate_position)~=1
        [~,ih,~] = intersect(hx,candidate_position,'rows');  
        if isempty(ih)~=1    % judge Repeat Sample    
            disp(['Sample repeat and delete it'])
            repeat_num= repeat_num+1;
        else
            for aa=1:size(candidate_position,1)
                candidate_fit(aa) = FUN(candidate_position(aa,:));
                candidate_position_size=candidate_position_size+1;
                Archive_FEs(NFE,:) = [NFE,candidate_fit(aa)];
                Archive_convergence(1,NFE) = min(Archive_FEs(1:NFE,2));
                NFE =  NFE +1;
                hx = [hx; candidate_position(aa,:)];  hf = [hf, candidate_fit(aa)];
            end
        end
    end

     %% Evolve the Loser sub-populations to generate the trial population
    LB = repmat((L_bound),NP,1);
    UB = repmat((U_bound),NP,1);
    P2=P(losers,:);
    fitness2=fitness(losers);
    U2=DE_update_rand(P2,NP/2,Dim,P(winners,:),F,CR,UB,LB,P);

    fitnessModel=Model_FUN(U2);
    [~,sidx] = sort(fitnessModel);
    candidate_position=[];
        dx2=min(sqrt(sum((repmat(U2(sidx(1),:),size(hx,1),1)-hx).^2,2)));
        if dx2 > dlta && fitnessModel(sidx(1))<hf(1)
            candidate_position=U2(sidx(1),:);
            candidate_position_size=candidate_position_size+1;
            candidate_fit = FUN(candidate_position);
            Archive_FEs(NFE,:) = [NFE,candidate_fit];
            Archive_convergence(1,NFE) = min(Archive_FEs(1:NFE,2));
            NFE =  NFE +1;
            hx = [hx; candidate_position];  hf = [hf, candidate_fit];
        end
        if candidate_position_size==0
            if  fitnessModelU1<hf(1) && (NFE/MaxNFE)>0.5
                candidate_position= bestU1;
                candidate_position_size=candidate_position_size+1;
                candidate_fit = FUN(candidate_position);
                Archive_FEs(NFE,:) = [NFE,candidate_fit];
                Archive_convergence(1,NFE) = min(Archive_FEs(1:NFE,2));
                NFE =  NFE +1;
                hx = [hx; candidate_position];  hf = [hf, candidate_fit];
            else
                for jj=1:NP/2
                    dx2=min(sqrt(sum((repmat(U2(sidx(jj),:),size(hx,1),1)-hx).^2,2)));
                    if  fitnessModel(sidx(jj))<fitness1(jj) && dx2>dlta
                        candidate_position=U2(sidx(jj),:);
                        candidate_position_size=candidate_position_size+1;
                        candidate_fit = FUN(candidate_position);
                        Archive_FEs(NFE,:) = [NFE,candidate_fit];
                        Archive_convergence(1,NFE) = min(Archive_FEs(1:NFE,2));
                        NFE =  NFE +1;
                        hx = [hx; candidate_position];  hf = [hf, candidate_fit];
                        break;
                    end
                end
            end
        end

   disp(['  Best fitness(Action ' num2str(Action)  ') = ' num2str(min(hf)) ' NFE=' num2str(NFE)]);
  %% Update database and display 
        if  sum(hf(end-candidate_position_size:end) < min(hf(1)))>=1
            action_success = 1;
           % R=1;
            R=5;
            flag_num=0;
            repeat_num= 0;
        else
            action_success = 0; 
            %R=0;
            R=-5;
            flag_num=flag_num+1;
        end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update agent
            State_Next = 2*Action+action_success-1;   
            temp = max(Q_Agent(State_Next,:)) ;
            Q_Agent(State,Action) = (1-alp)*Q_Agent(State, Action)+alp*(R+gamma*temp);
            State = State_Next;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% update agent

end

    function [offset] = cauchyMutation(scale_param)
        offset = scale_param * tan(pi * (rand() - 0.5));
    end

    function[randompop]=randomInitialization(Lbound,Ubound)
        randompop = repmat(Lbound,1,1)+(repmat(Ubound,1,1)-repmat(Lbound,1,1)).*lhsdesign(1,Dim);
    end

    function [ POP ] = initialize_pop(n,c,bu,bd)
        % Usage: [ POP ] = initialize_pop(n,c,bu,bd)
        % Input:
        % bu            -Upper Bound
        % bd            -Lower Bound
        % c             -No. of Decision Variables
        % n             -Population Scale
        %
        % Output:
        % POP           -Initial Population
        POP=lhsdesign(n,c).*(ones(n,1).*(bu-bd))+ones(n,1).*bd;    
    end
end

