% DEoperating consist of mutation and crossover
function [U] = DE_update_rand(P,NP,Dim,hisx,F,CR,UB,LB,center)
    for i=1:NP   
       k0=randi([1,NP]);
       while(k0==i)
           k0=randi([1,NP]);
       end
       P1=center(k0,:); 
       V(i,:)=P(i,:)+F.*(mean(hisx)-P(i,:));
      
       % bound
        for j=1:Dim
          if (V(i,j)>UB(i,j)||V(i,j)<LB(i,j))
             V(i,j)=LB(i,j)+rand*(UB(i,j)-LB(i,j));         
          end
        end

        % crossover
        jrand=randi([1,Dim]); 
        for j=1:Dim
            k3=rand;
            if(k3<=CR||j==jrand)
                U(i,j)=V(i,j);
            else
                U(i,j)=P(i,j);      
            end
        end
    end
end
