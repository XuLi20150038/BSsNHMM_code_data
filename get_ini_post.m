function fval = get_ini_post(Xtrn_L,Ytrn_L,Xtrn_U,label_ind,unlabel_ind,ini_prior,K)

[N1,dx]=size(Xtrn_L);
N2=size(Xtrn_U,1);
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

Y1=Ytrn_L;
X1=Xtrn_L;
X1_wan=[X1,ones(N1,1)];
X2=Xtrn_U;

Ro=rand(N1+N2,K);
deno=sum(Ro,2);
R=Ro./repmat(deno,1,K);

R1=R(label_ind,:);
sumR1=sum(R1);
R2=R(unlabel_ind,:);
sumR2=sum(R2);

a=a0+R1(1,:);

Aini=rand(K,K); % randomly initialized p(znk=1|znj_1=1,X)
Aini=Aini./repmat(sum(Aini,2),1,K);

for k=1:K
    R1k=R1(:,k);    
    N1k=sumR1(k);
    R2k=R2(:,k);
    N2k=sumR2(k);
    Rk=R(:,k);
    Wbp_ini_k=Wbp_ini{k};   
    Htrn_k=(1+exp(-X1_wan*Wbp_ini_k')).^(-1);    
    H_k=[Htrn_k,ones(N1,1)];
    S=size(H_k,2);  
    B(k,:)=b0+Aini(k,:)*sum(Rk(1:N1+N2-1));
    
    betak=beta0+N1k+N2k;
    mk=(R1k'*X1+R2k'*X2+beta0*m0)/betak;
    vk=v0+N1k+N2k;
    iWk=iW0+X1'.*repmat(R1k',dx,1)*X1+X2'.*repmat(R2k',dx,1)*X2+beta0*m0'*m0-betak*mk'*mk;
    Wk=inv(iWk); 
    
    ek=e0+S/2;
    fk=f0;
    pk=p0;
    qk=q0;  
    
    
       
   
    Gk=inv(pk/qk*H_k'.*repmat(R1k',S,1)*H_k+ek/fk*eye(S));
    gk=pk/qk*Gk*(H_k'.*repmat(R1k',S,1)*Y1);   
    
    ek=e0+(S)/2;
    fk=f0+(gk'*gk+trace(Gk))/2;
    
    pk=p0+N1k/2;    
    qk=q0+1/2*R1k'*(Y1-H_k*gk).^2+1/2*trace(H_k'.*repmat(R1k',S,1)*H_k*Gk);   
    
    beta(k)=betak;
    m(k,:)=mk;
    v(k)=vk;
    W{k}=Wk;     
    e(k)=ek;
    f(k)=fk;
    p(k)=pk;
    q(k)=qk;
    
    g(:,k)=gk;
    G{k}=Gk;    

end
ini_post.a=a;
ini_post.B=B;

ini_post.beta=beta;
ini_post.m=m;
ini_post.v=v;
ini_post.W=W;

ini_post.e=e;
ini_post.f=f;

ini_post.p=p;
ini_post.q=q;

ini_post.g=g;
ini_post.G=G;

fval=ini_post;
end

