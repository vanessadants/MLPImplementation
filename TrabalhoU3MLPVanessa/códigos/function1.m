%Author: Vanessa Dantas de Souto Costa
%email: vanessa.dantas796@gmail.com

%Primeira fun��o f(x)=sin(x)./x

%treinamento
passoTreinamento=0.1;

x=-4*pi:passoTreinamento:4*pi;
y=sin(x)./x;
x=x';
y=y';
in=x./max(abs(x)); %normalizar
out=y;

nce=3;
nuce=10;
TA=0.0014;

epMax=1000;
emqTarget=0.01;
percentrein=0.7;
alphaMomento=0.9;
face='LOGSIGMOIDE'; %op��es 'LOGSIGMOIDE' e 'TANGENTESIGMOIDE'
ini=[];

%carregar vari�veis
%load('ini.mat')
load('SaidasBackPropagation1.mat');

[Pesos,AtivacoesNos,saidas,EMQ,Epoca]=BackPropagation(in,out,nce,nuce,TA,face,ini,epMax,emqTarget,percentrein,alphaMomento);
EMQ=EMQ(1:Epoca);

%salvar variaveis
save('SaidasBackPropagation1.mat','Pesos','AtivacoesNos','EMQ','saidas','Epoca');

%plot da sa�da do treinamento
%figure();
%plot(saidas);

%validacao
passoValidacao=0.3;

xValidacao=-4*pi:passoValidacao:4*pi;
yValidacao=sin(xValidacao)./xValidacao;
xValidacao=xValidacao';
yValidacao=yValidacao';

inValidacao=xValidacao;
outValidacao=yValidacao;

[saidasValidacao]=MLPnetwork(inValidacao,AtivacoesNos,Pesos,face);


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
figure();
hold on
title('Compara��o entre fun��o real e sa�da MLP');
plot(xValidacao,yValidacao,'r');
plot(xValidacao,saidasValidacao,'b');
hold off

%Sa�dadaredeneural;
figure();
plot(xValidacao,saidasValidacao,'b');
title('Sa�da MLP');

%Sa�dadafun��omatem�tica;
figure();
plot(xValidacao,yValidacao,'r');
title('Fun��o Matem�tica');