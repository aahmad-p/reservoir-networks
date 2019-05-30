function prob=normaldistribution(x, mean, sigma)
%e=exp(1);
prob=(1/(sqrt(2*pi)*sigma))*exp((-((x-mean).^2)./(2*sigma^2)));
end