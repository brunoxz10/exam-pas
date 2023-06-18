# exam-pas

Este é o repositório do projeto **Exame PAS**. É uma ferramenta - destinada aos estudantes que irão prestar esse exame ou já iniciaram alguma das etapas - capaz de fornecer uma probabilidade de aprovação dado o curso desejado, se é cotista e preferencialmente notas das duas primeiras etapas.

Os dados estão originalmente presentes em *https://www.cebraspe.org.br/pas/subprogramas*, todos em formato .pdf e foram tratados por expressões regulares. 

Após a construção das *features*, um modelo XGBoost é ajustado. Os dados dos subprogramas 2019-2021 e 2020-2022 foram utlizados do seguinte modo: para treinamento, dados de ambos os subprogramas são considerados, mas o conjunto de dados de teste só apresenta observações de 2020-2022.

No final, o modelo treinado é salvo como .pickle e fica disponível para o script que monta a API por Flask. Por meio dessa API, é possível