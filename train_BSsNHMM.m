function fval=train_BSsNHMM(Xtrn,Ytrn,Xtrn_L,Ytrn_L,label_ind,K,ini_prior,ini_post,neuroNum,lr,maxIter,tol)

if isempty(maxIter)
    maxIter=200;
end
if isempty(tol)
    tol=1e-6;
end


a0=ini_prior.a0;
b0=ini_prior.b0;
beta0=ini_prior.beta0;
m0=ini_prior.m0;
v0=ini_prior.v0;
W0=ini_prior.W0;
iW0=inv(W0);

e0=ini_prior.e0;
f0=ini_prior.f0;

p0=ini_prior.p0;
q0=ini_prior.q0;
Wbp_ini=ini_prior.Wbp_ini;

a=ini_post.a;
B=ini_post.B;
beta=ini_post.beta;
m=ini_post.m;
v=ini_post.v;
W=ini_post.W;

g=ini_post.g;
G=ini_post.G;
e=ini_post.e;
f=ini_post.f;
p=ini_post.p;
q=ini_post.q;
Wbp=Wbp_ini;

[N,dx]=size(Xtrn);
NL=size(Xtrn_L,1);


alpha_hat=zeros(N,K);
beta_hat=zeros(N,K);
c=zeros(1,N);
pdf_d=zeros(N,K);
L_old=-1e-10;

for iter=1:maxIter
    iter;
    if mod(iter,30)==0
       lr=lr*0.1;
    end
   %% variational E-Estep
   
   %% forward-recursion
   E_PI=a/sum(a);
   E_A=B./repmat(sum(B,2),1,K);
   x1=Xtrn(1,:);
   for k=1:K
       mk=m(k,:);
       vk=v(k);
       betak=beta(k);
       Wk=W{k};
       Lk=(vk+1-dx)*betak/(1+betak)*Wk;
       pdf_x1(k)=get_t_pdf(x1,mk,Lk,vk+1-dx);
   end
         
   if ismember(1,label_ind)
       y1=Ytrn(1,:);
       x1=Xtrn(1,:);
       x1_wan=[x1,1];
        for k=1:K
            Wbp_k=Wbp{k};
            htrn1_k=(1+exp(-x1_wan*Wbp_k')).^(-1);
            h1_k=[htrn1_k,1];
            gk=g(:,k);
            Gk=G{k};
            pk=p(k);
            qk=q(k); 
            utemp=h1_k*gk;
            vtemp=qk/pk+h1_k*Gk*h1_k';
            pdf_y1(k)=get_normal(y1,utemp,vtemp);
        end
    else
        pdf_y1=ones(1,K);
   end
    
   pdf_d1=pdf_x1.*pdf_y1;
   
   pdf_d(1,:)=pdf_d1;
   
   c(1)=pdf_d1*E_PI';
   alpha_hat(1,:)=pdf_d1.*E_PI/c(1);
   
   for n=2:N
       xn=Xtrn(n,:);
       for k=1:K
           mk=m(k,:);
           vk=v(k);
           betak=beta(k);
           Wk=W{k};
           Lk=(vk+1-dx)*betak/(1+betak)*Wk;
           pdf_xn(k)=get_t_pdf(xn,mk,Lk,vk+1-dx);
       end
       if ismember(n,label_ind)
           yn=Ytrn(n,:);           
           xn_wan=[xn,1];
           for k=1:K
               Wbp_k=Wbp{k};
               htrnn_k=(1+exp(-xn_wan*Wbp_k')).^(-1);
               hn_k=[htrnn_k,1];
               gk=g(:,k);
               Gk=G{k};
               pk=p(k);
               qk=q(k);
               utemp=hn_k*gk;
               vtemp=qk/pk+hn_k*Gk*hn_k';
               pdf_yn(k)=get_normal(yn,utemp,vtemp);
           end
       else
           pdf_yn=ones(1,K);
       end
       pdf_dn=pdf_xn.*pdf_yn; 
       pdf_d(n,:)=pdf_dn;
      c(n)=pdf_dn*(alpha_hat(n-1,:)*E_A)';
      alpha_hat(n,:)=(pdf_dn.*(alpha_hat(n-1,:)*E_A))/c(n);
   end
   
   %% backward-recursion
   beta_hat(N,:)=1;
   for n=N-1:-1:1
       xn_plus_1=Xtrn(n+1,:);
       for k=1:K
           mk=m(k,:);
           vk=v(k);
           betak=beta(k);
           Wk=W{k};
           Lk=(vk+1-dx)*betak/(1+betak)*Wk;
           pdf_xn_plus_1(k)=get_t_pdf(xn_plus_1,mk,Lk,vk+1-dx);
       end
       if ismember(n+1,label_ind)
           yn_plus_1=Ytrn(n+1,:);
           xn_plus_1_wan=[xn_plus_1,1];
           for k=1:K
               Wbp_k=Wbp{k};
               htrnn_plus_1_k=(1+exp(-xn_plus_1_wan*Wbp_k')).^(-1);
               hn_plus_1_k=[htrnn_plus_1_k,1];
               gk=g(:,k);
               Gk=G{k};
               pk=p(k);
               qk=q(k);
               utemp=hn_plus_1_k*gk;
               vtemp=qk/pk+hn_plus_1_k*Gk*hn_plus_1_k';
               pdf_yn_plus_1(k)=get_normal(yn_plus_1,utemp,vtemp);
           end
       else
           pdf_yn_plus_1=ones(1,K);
       end
       pdf_dn_plus_1=pdf_xn_plus_1.*pdf_yn_plus_1;
       beta_hat(n,:)=(beta_hat(n+1,:).*pdf_dn_plus_1)*E_A'/c(n+1);
   end
   
   %% Convergence check
   L(iter)=sum(log(c));
   L_new=L(iter);
   if abs(L_new-L_old)/abs(L_old)<tol
       break;
   else
       L_old=L_new;
   end
   
   %% variational E-Mstep
   Ro=alpha_hat.*beta_hat;
   R=Ro./repmat(sum(Ro,2),1,K);
   X1_wan=[Xtrn_L,ones(NL,1)];
   R1=R(label_ind,:);
   sumR1=sum(R1);
   a=a0+R(1,:);
   xi_jk=zeros(K,K);
   for j=1:K
       
       Wbp_ini_j=Wbp_ini{j};
       Htrn_j=(1+exp(-X1_wan*Wbp_ini_j')).^(-1);
       H_j=[Htrn_j,ones(NL,1)];
       S=size(H_j,2);
       R1j=R1(:,j);
       N1k=sumR1(j);
       
       Rj=R(:,j);
       E_Aj=B(j,:)/sum(B(j,:));
       
       xi_nj=zeros(1,K); % xi_njk for fixed j
       xi_nj_sum=zeros(1,K);
       
       for n=2:N
           for k=1:K
               xi_nj(k)=E_Aj(k)*alpha_hat(n-1,j)*beta_hat(n,k)*pdf_d(n,k)/c(n);
           end
           xi_nj_sum=xi_nj_sum+xi_nj;
       end
       xi_jk(j,:)=xi_nj_sum;
       B(j,:)=xi_nj_sum+b0; 
       betaj=beta0+sum(Rj);
       mj=(Rj'*Xtrn+beta0*m0)/betaj;
       vj=v0+sum(Rj);
       iWj=iW0+Xtrn'.*repmat(Rj',dx,1)*Xtrn+beta0*m0'*m0-betaj*mj'*mj;
       Wj=inv(iWj);
       ej=e(j);
       fj=f(j);
       pj=p(j);
       qj=q(j);
       Gj=inv(pj/qj*H_j'.*repmat(R1j',S,1)*H_j+ej/fj*eye(S));
       gj=pj/qj*Gj*(H_j'.*repmat(R1j',S,1)*Ytrn_L);
       
       ej=e0+(S)/2;
       fj=f0+(gj'*gj+trace(Gj))/2;
       
       pj=p0+N1k/2;
       qj=q0+1/2*R1j'*(Ytrn_L-H_j*gj).^2+1/2*trace(H_j'.*repmat(R1j',S,1)*H_j*Gj);
       beta(j)=betaj;
       m(j,:)=mj;
       v(j)=vj;
       W{j}=Wj;
       e(j)=ej;
       f(j)=fj;
       p(j)=pj;
       q(j)=qj;
       
       g(:,j)=gj;
       G{j}=Gj;
          
   end
   %% variational M-step
        for k=1:K
            gk=g(:,k);
            Gk=G{k};  
            pk=p(k);
            qk=q(k);
            Wbp_k=Wbp{k};
            Htrn_k=(1+exp(-X1_wan*Wbp_k')).^(-1);
            H_k=[Htrn_k,ones(NL,1)];            
            g1=gk(1:neuroNum,:);
            G1=Gk(1:neuroNum,:);
            R1k=R1(:,k);
            R1k_ex=repmat(R1k,1,neuroNum);
            Kexi_k=(H_k*gk-Ytrn_L)*g1'+H_k*G1';
            Fai_k=R1k_ex.*Kexi_k.*(Htrn_k.*(1-Htrn_k));
            dW_k=-pk/qk*Fai_k'*X1_wan;
            Wbp_k=Wbp_k+lr*dW_k;
            Wbp{k}=Wbp_k;
        end
end
fval.a=a;
fval.B=B;
fval.beta=beta;
fval.m=m;
fval.v=v;
fval.W=W;

fval.g=g;
fval.G=G;
fval.e=e;
fval.f=f;
fval.p=p;
fval.q=q;
fval.Wbp=Wbp;
fval.alpha_hat_pre=alpha_hat(end,:);
fval.L=L;
end

   
   
   
   
   
   
   
   
   
   












