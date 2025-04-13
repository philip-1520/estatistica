"""
Encontra os coeficientes a e b de uma regressão linear y = a*x + b usando o método de Newton-Raphson para minimizar os quadrados dos desvios.

Parâmetros:
- dados: DataFrame contendo os dados
- x: nome da coluna independente
- y: nome da coluna dependente
- iteracoes: número de iterações para ajuste (default: 1000)

Retorna:
- Tupla (a, b) com os coeficientes arredondados
"""

import pandas as pd

def regressaolinear(dados: pd.DataFrame, x: str, y: str, iteracoes: int = 1000, precisao: int = 2):
    
    # Estima os valores iniciais de a e b utilizando os dois primeiros pontos.
    a = (dados.loc[1, y] - dados.loc[0, y])/(dados.loc[1, x] - dados.loc[0, x])
    b = dados.loc[0, y] - a*dados.loc[0, x]

    # Calcula a segunda derivada parcial da soma dos quadrados dos desvios em relação aos coeficientes a e b.
    dda = (2*dados[x]**2).sum()
    ddb = 2*len(dados[x])

    # O algoritmo aproxima as raízes da primeira derivada parcial de cada coeficiente utilizando o método de Newton-Raphson.
    for i in range(iteracoes):
    
        # Calcula a primeira derivada parcial da soma dos quadrados dos desvios em relação aos coeficientes a e b.
        da = ((-2*dados[x])*(dados[y] - a*dados[x] - b)).sum()
        db = (-2*(dados[y] - a*dados[x] - b)).sum()

        # Atualiza os valores de a e b para se aproximarem das raízes das funções.
        a = a - da/dda
        b = b - db/ddb

    # Retorna os valores dos coeficientes arredondados.
    return round(a, precisao), round(b, precisao)
    
