%Author: Vanessa Dantas de Souto Costa
%email: vanessa.dantas796@gmail.com

function saidas = EvaluateNetwork(Amostra, AtivacoesNos, Pesos,face)

nCamadas = length(AtivacoesNos);

AtivacoesNos{1} = Amostra;
for Camada = 2:nCamadas
    AtivacoesNos{Camada} = AtivacoesNos{Camada-1}*Pesos{Camada-1};
    
    if (Camada ~= nCamadas) %Because bias nodes don't have Pesos connected to previous Camada
        AtivacoesNos{Camada} = FuncaoAtivacao(AtivacoesNos{Camada},face);
        AtivacoesNos{Camada}(1) = 1;
    end
end

saidas = AtivacoesNos{end};

end