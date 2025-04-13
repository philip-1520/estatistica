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
from decimal import Decimal, getcontext

def regressaolinear(dados_originais: pd.DataFrame, x: str, y: str, iteracoes: int = 1000, precisao: int = 2) -> tuple[float, float]:

    # Define a precisão de 50 casas decimais.
    getcontext().prec = 50
    
    # Transforma os dados originais para decimal.
    dados = pd.DataFrame()
    dados[x] = dados_originais[x].apply(lambda x: Decimal(str(x)))
    dados[y] = dados_originais[y].apply(lambda y: Decimal(str(y)))

    # Declara a, b e uma constante dois como decimal.
    a = Decimal(1.0)
    b = Decimal(0.0)
    dois = Decimal(2.0)

    # Calcula a segunda derivada parcial da soma dos quadrados dos desvios em relação aos coeficientes a e b.
    dda = (dois*dados[x]**dois).sum()
    ddb = dois*len(dados[x])

    # O algoritmo aproxima as raízes da primeira derivada parcial de cada coeficiente utilizando o método de Newton-Raphson.
    for i in range(iteracoes):
    
        # Calcula a primeira derivada parcial da soma dos quadrados dos desvios em relação aos coeficientes a e b.
        da = ((-dois*dados[x])*(dados[y] - a*dados[x] - b)).sum()
        db = (-dois*(dados[y] - a*dados[x] - b)).sum()

        # Atualiza os valores de a e b para se aproximarem das raízes das funções.
        a = a - da/dda
        b = b - db/ddb

    # Retorna os valores dos coeficientes arredondados.
    return round(float(a), precisao), round(float(b), precisao)
