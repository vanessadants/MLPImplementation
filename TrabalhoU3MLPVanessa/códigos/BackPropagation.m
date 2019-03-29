%Author: Vanessa Dantas de Souto Costa
%email: vanessa.dantas796@gmail.com

%%%%%%%%%%%%%%%%%%%%%%Entradas%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%in --> entrada matriz 
%nAmostras = numero de entradas
%nSaidas = numero de saidas (exemplos de treinamento)

%ex.: Porta xor    entradas  saida esperada
%                   1 0         1
%                   1 1         0
%                   0 0         0
%                   0 1         1

%out --> sa�da esperada
%nce --> n�mero de camadas escondidas
%nuce --> n�mero de neur�nios na camada escondida. 
%Obs.: para a �ltima camada (camada de sa�da), devemos ter apenas um
%neur�nio e fun��o de ativa��o linear
%TA = taxa de aprendizagem
%face --> fun��o de ativa��o da camada escondida
%ini --> matriz de pesos iniciais(utilizada para agilizar treinamento). Se
%passada vazia, devemos inici�-la aleatoriamente

%epMax --> n�mero de epocas para treinamento
%emqTarget --> erro m�dio quadr�tico m�nimo desejado
%percentrein --> percentual de divis�o do grupo de treinamento em
%"treinamento" e "valida��o", sugerido algo entre 70% e 80%

%alphaMomento --> constante menor que 1, utilizada para c�lculo do momento

%%%%%%%%%%%%%%%%%%%%%%Sa�das%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Pesos --> vetor de pesos
%AtivacoesNos --> entradas de cada neur�nio
%saidas --> saida para plot
%EMQ --> erro m�dio quadr�tico de cada �poca
%Epoca --> n�mero de itera��es do BackPropagation

%%%%%%%%%%%%%%%%Fun��o que executa o treinamento%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Pesos,AtivacoesNos,saidas,EMQ,Epoca]=BackPropagation(in,out,nce,nuce,TA,face,ini,epMax,emqTarget,percentrein, alphaMomento)

% Separar Grupo de treino do grupo de validacao 
     %embaralhar colunas
     [N p]=size(in);
     auxrand=randperm(N);
     in=in(auxrand,:);%desse modo trocamos a ordem das linhas
     out=out(auxrand,:);
    
     %devemos pegar percentrein para treinamento e o restante para
     %validacao
     inTreino =in(1:floor(percentrein*N),:);
     outTreino =out(1:floor(percentrein*N),:);
     inValidacao =in((floor(percentrein*N)+1):end,:);
     outValidacao =out((floor(percentrein*N)+1):end,:);
     
     in=inTreino;
     out=outTreino;
     
%inicializar saidas
saidas=zeros(length(in(:,1)),1);
saidasValidacaoTreinamento=zeros(length(inValidacao(:,1)),1);

%determina o n�mero de neur�nios de cada camada escondida
nNeuroniosCadaCamadaEscondida = ones(1,nce)*nuce;
%calcula o nAmostras e o nSaidas com base na resposta desejada
[nAmostras nSaidas] = size(out);

%calcula o n�mero de n�s baseado na entrada
nNosEntrada = length(in(1,:));

%calcula nCamadas e o nNosPorCamada
nCamadas = 2 + length(nNeuroniosCadaCamadaEscondida);
nNosPorCamada = [nNosEntrada nNeuroniosCadaCamadaEscondida nSaidas];

%adicionar o Bias
nNosPorCamada(1:end-1) = nNosPorCamada(1:end-1) + 1;
in = [ones(length(in(:,1)),1) in];
inValidacao=[ones(length(inValidacao(:,1)),1) inValidacao];
%Pesos conectando nos de bias com camadas anteriores s�o
%desnecessariosare useless, mas para simplificar o c�digo e torn�-lo mais
%r�pido consideramos PesosDelta = cell(1,nCamadas);
Pesos = cell(1, nCamadas); 
PesosDelta = cell(1, nCamadas);

for i = 1:length(Pesos)-1
    Pesos{i} = 2*rand(nNosPorCamada(i), nNosPorCamada(i+1))-1; 
    Pesos{i}(:,1) = 0; %Pesos nos do bias com camada anterior (redundante)
    PesosDelta{i} = zeros(nNosPorCamada(i), nNosPorCamada(i+1));
end

%Pesos virtuais para nos de saida
Pesos{end} = ones(nNosPorCamada(end), 1);

%caso seja passado um parametro para iniciar os pesos
if ~isempty(ini)
    Pesos=ini;
end;


AtivacoesNos = cell(1, nCamadas);
for i = 1:length(AtivacoesNos)
    AtivacoesNos{i} = zeros(1, nNosPorCamada(i));
end
%� necess�rios para o treinamento do Backpropagation de tr�s pra frente
NosErrosPropagados = AtivacoesNos; 

emqTargetComprido = 0; %verificar se o erro emqTarget foi alcan�ado

%inicializa��o do vetor de erros por �poca
EMQ = -1 * ones(1,epMax);

%Backpropagation e atualiza��o de pesos delta
PesosDeltaAntigosMomento = PesosDelta;
for Epoca = 1:epMax
    for Amostra = 1:length(in(:,1))
        AtivacoesNos{1} = in(Amostra,:);
        for Camada = 2:nCamadas
            AtivacoesNos{Camada} = AtivacoesNos{Camada-1}*Pesos{Camada-1};
            %AtivacoesNos{Camada} = FuncaoAtivacao(AtivacoesNos{Camada},face);
            %Porque os n�s do Bias n�o tem Pesos conectados com camadas anteriores
            if (Camada ~= nCamadas) 
                AtivacoesNos{Camada}(1) = 1;
                AtivacoesNos{Camada} = FuncaoAtivacao(AtivacoesNos{Camada},face);
            end
        end
        % Armazenamento dos erros passados para tr�s
        % (As gradiente of the bias nodes are zeros, they won't contribute to previous Camada errors nor PesosDelta)
        NosErrosPropagados{nCamadas} =  out(Amostra,:)-AtivacoesNos{nCamadas};
        for Camada = nCamadas-1:-1:1
            if(Camada~=(nCamadas-1))
                gradiente=FuncaoAtivacaoDerivada(AtivacoesNos{Camada+1},face);
            else
                gradiente=1;
            end
            for node=1:length(NosErrosPropagados{Camada}) % For all the Nodes in current Camada
                NosErrosPropagados{Camada}(node) =  sum( NosErrosPropagados{Camada+1} .* gradiente .* Pesos{Camada}(node,:) );
            end
        end
        % Calculo dos pesos delta passados para tr�s (antes da multiplica��o pela taxa de aprendizagem) 
        for Camada = nCamadas:-1:2
            if(Camada~=nCamadas)
                derivative = FuncaoAtivacaoDerivada(AtivacoesNos{Camada},face);    
            else
                derivative=1;
            end    
            PesosDelta{Camada-1} = PesosDelta{Camada-1} + AtivacoesNos{Camada-1}' * (NosErrosPropagados{Camada} .* derivative);
        end
    end
   
    
    %Aplicar Momento
 	for Camada = 1:nCamadas
     	PesosDelta{Camada} = TA*PesosDelta{Camada} + alphaMomento*PesosDeltaAntigosMomento{Camada}; 
     end
 	PesosDeltaAntigosMomento = PesosDelta;
    
     % Atualiza��o dos pesos
    for Camada = 1:nCamadas-1
        Pesos{Camada} = Pesos{Camada} + PesosDelta{Camada};
    end
    
    % Resetar PesosDelta para Zeros
    for Camada = 1:length(PesosDelta)
        PesosDelta{Camada} = 0 * PesosDelta{Camada};
    end
    
    
    for Amostra = 1:length(in(:,1))
        saidas(Amostra) = EvaluateNetwork(in(Amostra,:), AtivacoesNos, Pesos,face);
    end
    
    %Calcular EMQ da epoca
    %Validacao do Treinamento
    for Amostra = 1:length(inValidacao(:,1))
        saidasValidacaoTreinamento(Amostra)=EvaluateNetwork(inValidacao(Amostra,:), AtivacoesNos, Pesos,face);
    end
    
    
    EMQ(Epoca)= sum((saidasValidacaoTreinamento-outValidacao).^2)/(length(inValidacao(:,1)));
    if (EMQ(Epoca) < emqTarget)
        emqTargetComprido = 1;
    end
    
    
    display([int2str(Epoca) ' epocas de um total de ' int2str(epMax) ' epocas m�ximas. EMQ = ' num2str(EMQ(Epoca)) ' Taxa de Aprendizagem = ' ...
        num2str(TA) '.']);     
   
    %Caso tenhamos atingido o erro desejado, podemos parar o programa
    if (emqTargetComprido)
        EMQ=EMQ(1:Epoca);
        break;
    end
    
    %salvar variaveis
    ini=Pesos;
    save('SaidasBackPropagation.mat','Pesos','AtivacoesNos','EMQ','saidas','Epoca');
    save('ini.mat','ini');
end
    
   

end