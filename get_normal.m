function fval=get_normal(X,u,V)
[n,d]=size(X);
X=X-repmat(u,n,1);% centering

V=V+realmin*eye(d);

A=(2*pi)^(d/2)*sqrt(abs(det(V))); % denominator

B=sum((X/V).*X, 2);

fval=exp(-1/2*B)/A;
