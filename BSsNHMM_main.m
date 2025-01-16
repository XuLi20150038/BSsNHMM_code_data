clc,clear
close all


load data_numerical_BSsNHMM


mY=mean(y);
stdY=std(y);
X=zscore(X);
Y=zscore(y);
% Y=y;

dx=size(X,2);
dy=size(Y,2);

K=4;

N=length(Y);
Ntst=2000;

Nval=2000;
Xval=X(N-Ntst-Nval+1:N-Ntst,:);
Yval=Y(N-Ntst-Nval+1:N-Ntst,:);
Xtrn=X(1:N-Ntst-Nval,:);
Ytrn=Y(1:N-Ntst-Nval,:);

% Xtrn=X(6001:6750,:);
% Ytrn=Y(6001:6750,:);
% Xtrn=X(1:N-Ntst,:);
% Ytrn=Y(1:N-Ntst,:);

Xtst=X(N-Ntst+1:N,:);
Ytst=Y(N-Ntst+1:N,:);

label_rate=0.1;

Ntrn=size(Xtrn,1);

label_ind=round(linspace(1,Ntrn,round(label_rate*Ntrn)));
if label_ind(1)<1
    label_ind(1)=1;
end

Ntrn_label=size(label_ind,2);
unlabel_ind=setdiff(1:Ntrn,label_ind);
Xtrn_L=Xtrn(label_ind,:);
Xtrn_U=Xtrn(unlabel_ind,:);
Ytrn_L=Ytrn(label_ind,:);

lr=0.0001;
%% hyperparameter

ini_prior.a0=1;
ini_prior.b0=1;
ini_prior.m0=0*ones(1,dx);
ini_prior.beta0=1;
ini_prior.v0=dx+1;
ini_prior.W0=100*eye(dx);
ini_prior.e0=1;
ini_prior.f0=1;
ini_prior.p0=1;
ini_prior.q0=1;


neuroNum=200; %Number of hidden layer neurons
Wbp_ini_k=randn(neuroNum,dx+1);% random weight for input layer;
for k=1:K   
    
    Wbp_ini{k}=Wbp_ini_k;
end
ini_prior.Wbp_ini=Wbp_ini;

%% Initialize the posterior distribution
ini_post=get_ini_post(Xtrn_L,Ytrn_L,Xtrn_U,label_ind,unlabel_ind,ini_prior,K);

%% Train
s1=tic;
fval=train_BSsNHMM(Xtrn,Ytrn,Xtrn_L,Ytrn_L,label_ind,K,ini_prior,ini_post,neuroNum,lr,[],[]);
CPT=toc(s1);

% figure;
% plot(fval.L);
% xlabel('iteration step')
% ylabel('log-likelihood function')


%% Prediction
beta=fval.beta;
m=fval.m;
v=fval.v;
W=fval.W;
B=fval.B;
Wbp=fval.Wbp;
g=fval.g;
alpha_hat_pre=fval.alpha_hat_pre;

E_A=B./repmat(sum(B,2),1,K);




%% VAL


Yval_pre=zeros(Nval,1);
for i=1:Nval
    xi=Xval(i,:);
    xi_wan=[xi,1];
    for k=1:K
        mk=m(k,:);
        vk=v(k);
        betak=beta(k);
        Wk=W{k};
        Lk=(vk+1-dx)*betak/(1+betak)*Wk;  
        pdf_xik=get_t_pdf(xi,mk,Lk,vk+1-dx);
        pdf_dik=pdf_xik;
        alpha_hat_inter(k)=pdf_dik*alpha_hat_pre*E_A(:,k);  
        
        gk=g(:,k);
        Wbp_k=Wbp{k};
        Hi_k=(1+exp(-xi_wan*Wbp_k')).^(-1);
        Hi_wan_k=[Hi_k 1];
        yi_pre(1,k)=Hi_wan_k*gk;      
        
    end
    alpha_hat=alpha_hat_inter/sum(alpha_hat_inter);   
    alpha_hat_pre=alpha_hat;
    
    Yval_pre(i,:)=alpha_hat*yi_pre';  
end
Yval_pre=Yval_pre*stdY+mY;
Yval=Yval*stdY+mY;

RMSE_val=sqrt((Yval_pre-Yval)'*(Yval_pre-Yval)/Nval);
R2_val=1-sum((Yval-Yval_pre).^2)/sum((Yval-mean(Yval)).^2);


fontsize=14;
figure;hold on
plot(Yval,'r-','linewidth',3)
plot(Yval_pre,'b-','linewidth',2)
% plot(Ytst_n,'g--','linewidth',1)
% h=legend('real value','predicted value','noisy value','fontsize',fontsize);
h=legend('real value','predicted value','fontsize',fontsize);
xlabel('val sample number','fontsize',fontsize);
ylabel('y');
set(h,'box','off')
set(gca,'fontsize',fontsize);
set (gcf,'Position',[403   246   700  370], 'color','w')
box on;
figure;hold on
plot(Yval,Yval_pre,'r*')
xlabel('true value');
ylabel('predicted value');
set(gca,'fontsize',14);
box on;

%% TST


Ypre=zeros(Ntst,1);
for j=1:Ntst
    xj=Xtst(j,:);
    xj_wan=[xj,1];
    for k=1:K
        mk=m(k,:);
        vk=v(k);
        betak=beta(k);
        Wk=W{k};
        Lk=(vk+1-dx)*betak/(1+betak)*Wk;  
        pdf_xjk=get_t_pdf(xj,mk,Lk,vk+1-dx);
        pdf_djk=pdf_xjk;
        alpha_hat_inter(k)=pdf_djk*alpha_hat_pre*E_A(:,k);  
        
        gk=g(:,k);        
        Wbp_k=Wbp{k};        
        Hj_k=(1+exp(-xj_wan*Wbp_k')).^(-1);
        Hj_wan_k=[Hj_k 1];
        yj_pre(1,k)=Hj_wan_k*gk;     
        
    end
    alpha_hat=alpha_hat_inter/sum(alpha_hat_inter); 
    alpha_hat_all(j,:)=alpha_hat;
    alpha_hat_pre=alpha_hat;
    
    Ytst_pre(j,:)=alpha_hat*yj_pre';  
end
Ytst_pre=Ytst_pre*stdY+mY;
Ytst=Ytst*stdY+mY;


RMSE_tst=sqrt((Ytst_pre-Ytst)'*(Ytst_pre-Ytst)/Ntst);
R2_tst=1-sum((Ytst-Ytst_pre).^2)/sum((Ytst-mean(Ytst)).^2);

alpha_hat_SsNHMM=alpha_hat_all;
save alpha_hat_SsNHMM alpha_hat_SsNHMM


fontsize=16;
figure;hold on
plot(Ytst,'ro-','linewidth',1)
plot(Ytst_pre,'b-','linewidth',2)
% plot(Ytst_n,'g--','linewidth',1)
% h=legend('real value','predicted value','noisy value','fontsize',fontsize);
h=legend('real value','predicted value','fontsize',fontsize);
xlabel('test sample number','fontsize',fontsize);
ylabel('y');
set(h,'box','off')
set(gca,'FontName','Times New Roman','LooseInset', [0,0,0.01,0.01],'LineWidth',0.8,'fontsize',fontsize);
set(gcf,'Units','centimeter','Position',[5 5 28 9],'color','w');
box on;
figure;hold on
plot(Ytst,Ytst_pre,'r*')
xlabel('true value');
ylabel('predicted value');
set(gca,'FontName','Times New Roman','LooseInset', [0,0,0.01,0.01],'LineWidth',0.8,'fontsize',fontsize);
set(gcf,'color','w');
box on;

% save NHMM E_A



