%Author: Vanessa Dantas de Souto Costa
%email: vanessa.dantas796@gmail.com


%in --> entrada matriz 
%Pesos --> vetor de pesos
%AtivacoesNos --> entradas de cada neurônio
%face -->Funcao ativação

%Função para validacao dado que a rede ja foi treinada

function [saidas]=MLPnetwork(in,AtivacoesNos,Pesos,face)
    in = [ones(length(in(:,1)),1) in];
    saidas=zeros(length(in(:,1)),1);
    for Amostra = 1:length(in(:,1))
            saidas(Amostra) = EvaluateNetwork(in(Amostra,:), AtivacoesNos, Pesos,face);
    end
end