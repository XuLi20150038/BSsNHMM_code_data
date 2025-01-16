function fval=get_t_pdf(X,mu,Lambda,v)

[N,D]=size(X);
ln_Item1=gammaln(v/2+D/2)-gammaln(v/2);
Item1=exp(ln_Item1);
Item2=(det(Lambda))^0.5;
Item3=(v*pi)^(D/2);

X0=X-repmat(mu,N,1);
delta2=sum((X0*Lambda).*X0, 2);

fval=Item1*Item2/Item3*(1+delta2/v).^(-v/2-D/2);


