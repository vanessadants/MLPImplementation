%Author: Vanessa Dantas de Souto Costa
%email: vanessa.dantas796@gmail.com

%Primeira função f(x)=sin(x)./x

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
face='LOGSIGMOIDE'; %opções 'LOGSIGMOIDE' e 'TANGENTESIGMOIDE'
ini=[];

%carregar variáveis
%load('ini.mat')
load('SaidasBackPropagation1.mat');

[Pesos,AtivacoesNos,saidas,EMQ,Epoca]=BackPropagation(in,out,nce,nuce,TA,face,ini,epMax,emqTarget,percentrein,alphaMomento);
EMQ=EMQ(1:Epoca);

%salvar variaveis
save('SaidasBackPropagation1.mat','Pesos','AtivacoesNos','EMQ','saidas','Epoca');

%plot da saída do treinamento
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


%plot das solicitações feitas pelo professor

%ErroQuadráticoporépoca;
figure();
plot(EMQ);
title('Erro médio quadrático por Epoca');

%Errodecomparação;
erroComp=saidasValidacao-outValidacao;
figure();
plot(erroComp);
title('Erro de comparação entre a função real e a saída da MLP');

%Gráficodecomparação;
figure();
hold on
title('Comparação entre função real e saída MLP');
plot(xValidacao,yValidacao,'r');
plot(xValidacao,saidasValidacao,'b');
hold off

%Saídadaredeneural;
figure();
plot(xValidacao,saidasValidacao,'b');
title('Saída MLP');

%Saídadafunçãomatemática;
figure();
plot(xValidacao,yValidacao,'r');
title('Função Matemática');