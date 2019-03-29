%Author: Vanessa Dantas de Souto Costa
%email: vanessa.dantas796@gmail.com

%Quinta função f(x,y)= -(y+47).*sin( sqrt(abs(x./2+ y + 47 )) ) - x.*sin(sqrt(abs(x- y - 47))); 

%treinamento
passoTreinamento=100;

[x,y]=meshgrid(-1000:passoTreinamento:1000);
z = -(y+47).*sin( sqrt(abs(x./2+ y + 47 )) ) - x.*sin(sqrt(abs(x- y - 47)));
in=[x(:)./max(max(abs(x(:)))) y(:)./max(max(abs(y(:))))];
out=z(:);

nce=4;
nuce=25;
TA=0.00005;

epMax=500;
emqTarget=1.7;
percentrein=0.7;
alphaMomento=0.9;
face='LOGSIGMOIDE'; %opções 'LOGSIGMOIDE' e 'TANGENTESIGMOIDE'
ini=[];

%carregar variáveis
%load('ini.mat')
load('SaidasBackPropagation5.mat');

[Pesos,AtivacoesNos,saidas,EMQ,Epoca]=BackPropagation(in,out,nce,nuce,TA,face,ini,epMax,emqTarget,percentrein,alphaMomento);
EMQ=EMQ(1:Epoca);

%salvar variaveis
save('SaidasBackPropagation5.mat','Pesos','AtivacoesNos','EMQ','saidas','Epoca');

%plot da saída do treinamento
%figure();
%plot(saidas);

%validacao
passoValidacao=7;

[xValidacao,yValidacao]=meshgrid(-1000:passoValidacao:1000);
zValidacao=-(yValidacao+47).*sin( sqrt(abs(xValidacao./2+ yValidacao + 47 )) ) - xValidacao.*sin(sqrt(abs(xValidacao- yValidacao - 47)));
inValidacao=[xValidacao(:) yValidacao(:)];
outValidacao=zValidacao(:);

[saidasValidacao]=MLPnetwork(inValidacao,AtivacoesNos,Pesos,face);
SaidaPlot=vec2mat(saidasValidacao,sqrt(length(saidasValidacao)));

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
%figure();
%hold on
%mesh(xValidacao,yValidacao,zValidacao);
%mesh(xValidacao,yValidacao,SaidaPlot);
%title('Comparação entre função real e saída MLP');
%hold off


%Saídadaredeneural;
figure();
mesh(xValidacao,yValidacao,SaidaPlot);
title('Saída MLP');

%Saídadafunçãomatemática;
figure();
mesh(xValidacao,yValidacao,zValidacao);
title('Função Matemática');