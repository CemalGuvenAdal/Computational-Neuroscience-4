question = input("Enter Question Number",'s')
CemalGuven_Adal_21703986_hw4(question)
function CemalGuven_Adal_21703986_hw4(question)
clc
close all

switch question
    case '1'
	disp('1')
        % a_________
data=load('hw4_data1.mat');
face=data.faces();
meanface=mean(face,1);
face2=face-meanface;
[eigfaces,score,eigenvalues]=pca(face2);
yirmibeseig=eigfaces(:,1:25);
%25 res
figure()
dispImArray(yirmibeseig.',32)
title('First 25 Principal Components');

var1=eigenvalues(1:100)/sum(eigenvalues);
figure()
plot(var1)
title('Total Explained Variance by PC')
xlabel('PC');
ylabel('Variance');
size(meanface)
% b----------------
eigten=eigfaces(:,1:10);
eigyirmibes=eigfaces(:,1:25);
eigelli=eigfaces(:,1:50);
innerproductten = face2*eigten ;
tendisp = innerproductten* eigten.' + meanface;
figure()
dispImArray(tendisp(1:36,:),32)
title('Images Reconstructed by 10 PC')
innerproductyirmibes = face2*eigyirmibes ;
yirmibesdisp = innerproductyirmibes* eigyirmibes.' + meanface;
figure()
dispImArray(yirmibesdisp(1:36,:),32);
title('Images Reconstructed by 25 PC')
innerproductelli = face2*eigelli ;
ellidisp = innerproductelli* eigelli.' + meanface;
figure()
dispImArray(ellidisp(1:36,:),32);
title('Images Reconstructed by 50 PC')

figure()
dispImArray(face(1:36,:),32);
title('original faces');

% mse var of ten
aten=sum((tendisp-face).*(tendisp-face),2)/1024;
mseTen=mean(aten)
varTen=std(aten)
% mse var of twentyfive
atwenty=sum((yirmibesdisp-face).*(yirmibesdisp-face),2)/1024;
mseTwentyfive=mean(atwenty)
varTwentyfive=std(atwenty)
% mse var of fifty
afifty=sum((ellidisp-face).*(ellidisp-face),2)/1024;
msefifty=mean(afifty)
varfifty=std(afifty)


% part c
% 10
[tenIC, tenA, tenW] = fastica(face, 'lastEig', 50,'numOfIc', 10,'maxNumIterations', 5000);
figure(); sgtitle('IC 10');
dispImArray(tenIC, 32);
colorbar();
reconc=tenA*tenIC;
dispImArray(reconc(1:36,:),32);
% 25
[twentIC, twentA, twentW] = fastica(face, 'lastEig', 50,'numOfIc', 25,'maxNumIterations', 5000);
figure(); 
sgtitle('IC 25');
dispImArray(twentIC, 32);
colorbar();
rectwentc=twentA*twentIC;
dispImArray(rectwentc(1:36,:),32);
% 50
[elliIC, elliA, elliW] = fastica(face, 'lastEig', 50,'numOfIc', 50,'maxNumIterations', 5000);
figure(); 
sgtitle('IC 50');
dispImArray(elliIC, 32);
colorbar();
recellic=elliA*elliIC;
dispImArray(recellic(1:36,:),32);
% mse,var 10
atenc=sum((reconc-face).*(reconc-face),2)/1024;
mseTen=mean(atenc)
varTen=std(atenc)

% mse,var 25
atwentc=sum((rectwentc-face).*(rectwentc-face),2)/1024;
msefifty=mean(atwentc)
varfifty=std(atwentc)
% mse,var 50
afiftyc=sum((recellic-face).*(recellic-face),2)/1024;
msefifty=mean(afiftyc)
varfifty=std(afiftyc)


% partd
% 10
face3=face+abs(min(face,[],'all'));
[W,H]=nnmf(face3.',10,'algorithm', 'mul');
figure()
dispImArray(W.',32);
title('W=10');
recon1=(W*H)-abs(min(face,[],'all'));
figure()
dispImArray(recon1(:,1:36).',32);
title('Non Negative Matrix Reconstruction of Images by 10');
% 25
[W25,H25]=nnmf(face3.',25,'algorithm', 'mul');
figure()
dispImArray(W25.',32);
title('W=25');
recon2=(W25*H25)-abs(min(face,[],'all'));
figure()
dispImArray(recon2(:,1:36).',32);
title('Non Negative Matrix Reconstruction of Images by 25');
% 50
[W50,H50]=nnmf(face3.',50,'algorithm', 'mul');
figure()
dispImArray(W50.',32);
title('W=50');
recon5=(W50*H50)-abs(min(face,[],'all'));
figure()
dispImArray(recon5(:,1:36).',32);
title('Non Negative Matrix Reconstruction of Images by 50');

% mse,var 10
atend=sum((recon1.'-face).*(recon1.'-face),2)/1024;
mseTen=mean(atend)
varTen=std(atend)
% mse,var 25
atwentd=sum((recon2.'-face).*(recon2.'-face),2)/1024;
msetwentyfive=mean(atwentd)
vartwentyfive=std(atwentd)
% mse,var 50
afiftd=sum((recon5.'-face).*(recon5.'-face),2)/1024;
msefifty=mean(afiftd)
varfifty=std(afiftd)
	y = myfunction(3,5)
    case '2'
	disp('2')
        % QUESTION2-------------------------------------
% Part a
u=[-10:10];
A=1
sigma=1
x=[-20:0.1:20];
figure()
for i=1:21
    f=A*exp(-((x-u(i)).^2)/(2*sigma));
    plot(x,f);
    hold on
end
hold off
title('Tuning Curves');
xlabel('Stimulus');
ylabel('Response');

figure()
f1=A*exp(-((-1-u).^2)/(2*sigma));
plot(u,f1)
title('Preffered Stimulus Value');
ylabel('Response');
xlabel('Preffered Stimulus');
% b    

r=unifrnd(-5,5,1,200);
noise=0.05*randn(1,200);
noisedr=r+noise;
estimate=zeros(200,1);
for j=1:200
  [M,I] =min(abs(noisedr(j)-u));
estimate(j)=u(I);
end
figure()
scatter(1:200,estimate)
hold on
scatter(1:200,r)
title('Actual Values vs Winner Take all decoder Estimated Values');
xlabel('Trials');
ylabel('Stimulus');
var5=estimate-r;
estimateerror=mean(var5,'all')
estimatestd=std(var5(:))
% part c
r=unifrnd(-5,5,1,200);
noise=0.05*randn(1,200);
noisedr=r+noise;
estimate=zeros(200,1);
for j=1:200
  [M,I] =min(abs(noisedr(j)-u));
estimate(j)=u(I);
end
figure()
scatter(1:200,noisedr)
hold on
scatter(1:200,r)
title('Actual Values vs ML Estimated Values');
xlabel('Trials');
ylabel('Stimulus');
var6=estimate-noisedr;
estimateerror=mean(var6,'all')
estimatestd=std(var6(:))
% part d
x=[-5:0.01:5];
prior=(1/(sqrt(2*pi)*2.5))*exp((-x.*x)/(2*2.5*2.5));
mapestimate=zeros(1,200);
for j=1:200
 mapestimate(j)= decoder(prior,r(j));

end
figure()
scatter(1:200,mapestimate);
hold on
scatter(1:200,r);
legend('mapestimate','originaltrial')
var7=abs(mapestimate-r);
estimateerror=mean(var7,'all');
estimatestd=std(var7(:));
% part e
u=[-10:10];

sigma=[0.1,0.2,0.5,1,2,5];
for j=1:6
Mlestimate=zeros(1,200);
for i=1:200
    f5=exp(-((noisedr(i)-u).^2)/(2*sigma(j)*sigma(j)));
    [M,I]=max(f5(:));
    Mlestimate(1,i)=u(I);
    
end
disp("------------");
disp(sigma(j))
mean(abs(Mlestimate-r))
std(abs(Mlestimate-r))
disp("------------------");

end

   
end

end

function y = myfunction(a,b)
y = a+b;
end




function mapdecoder=decoder(prior,u)
x=[-5:0.01:5];
likelihood=exp((-(x-u).*(x-u))/2);
posterior=likelihood.*prior;
[W,I]=max(posterior);
mapdecoder=x(I);
end

