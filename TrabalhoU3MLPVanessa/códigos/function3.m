%Author: Vanessa Dantas de Souto Costa
%email: vanessa.dantas796@gmail.com

%Terceira fun��o f(x,y)=( y.*cos(x)+x.*sin(y))./(x.*y);

%treinamento
passoTreinamento=0.9;

[x,y]=meshgrid(0.1:passoTreinamento:2*pi);
z = (y.*cos(x)+x.*sin(y))./(x.*y); 
in=[x(:)./max(max(abs(x(:)))) y(:)./max(max(abs(y(:))))];
out=z(:);

nce=1;
nuce=25;
TA=0.0005;

epMax=700;
emqTarget=0.01;
percentrein=0.7;
alphaMomento=0.9;
face='LOGSIGMOIDE'; %op��es 'LOGSIGMOIDE' e 'TANGENTESIGMOIDE'
ini=[];

%carregar vari�veis
%load('ini.mat')
load('SaidasBackPropagation3.mat');

[Pesos,AtivacoesNos,saidas,EMQ,Epoca]=BackPropagation(in,out,nce,nuce,TA,face,ini,epMax,emqTarget,percentrein,alphaMomento);
EMQ=EMQ(1:Epoca);

%salvar variaveis
save('SaidasBackPropagation3.mat','Pesos','AtivacoesNos','EMQ','saidas','Epoca');

%validacao
passoValidacao=0.09;

[xValidacao,yValidacao]=meshgrid(0.1:passoValidacao:2*pi);
zValidacao = ( yValidacao.*cos(xValidacao)+xValidacao.*sin(yValidacao))./(xValidacao.*yValidacao);
inValidacao=[xValidacao(:) yValidacao(:)];
outValidacao=zValidacao(:);

[saidasValidacao]=MLPnetwork(inValidacao,AtivacoesNos,Pesos,face);
SaidaPlot=vec2mat(saidasValidacao,sqrt(length(saidasValidacao)));

%plot das solicita��es feitas pelo professor

%ErroQuadr�ticopor�poca;
figure();
plot(EMQ);
title('Erro m�dio quadr�tico por Epoca');

%Errodecompara��o;
erroComp=saidasValidacao-outValidacao;
figure();
plot(erroComp);
title('Erro de compara��o entre a fun��o real e a sa�da da MLP');

%Gr�ficodecompara��o;
%figure();
%hold on
%mesh(xValidacao,yValidacao,zValidacao);
%mesh(xValidacao,yValidacao,SaidaPlot);
%title('Compara��o entre fun��o real e sa�da MLP');
%hold off

%Sa�dadaredeneural;
figure();
mesh(xValidacao,yValidacao,SaidaPlot);
title('Sa�da MLP');


%Sa�dadafun��omatem�tica;
figure();
mesh(xValidacao,yValidacao,zValidacao);
title('Fun��o Matem�tica');