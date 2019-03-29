%Author: Vanessa Dantas de Souto Costa
%email: vanessa.dantas796@gmail.com

%Quinta fun��o f(x,y)= -(y+47).*sin( sqrt(abs(x./2+ y + 47 )) ) - x.*sin(sqrt(abs(x- y - 47))); 

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
face='LOGSIGMOIDE'; %op��es 'LOGSIGMOIDE' e 'TANGENTESIGMOIDE'
ini=[];

%carregar vari�veis
%load('ini.mat')
load('SaidasBackPropagation5.mat');

[Pesos,AtivacoesNos,saidas,EMQ,Epoca]=BackPropagation(in,out,nce,nuce,TA,face,ini,epMax,emqTarget,percentrein,alphaMomento);
EMQ=EMQ(1:Epoca);

%salvar variaveis
save('SaidasBackPropagation5.mat','Pesos','AtivacoesNos','EMQ','saidas','Epoca');

%plot da sa�da do treinamento
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