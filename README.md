# Trabalho Prático 2 de Inteligência Artificial

## Passo 1: Criação das instâncias

### Instrodução

Neste capítulo, tratamos da criação de instâncias para treino de uma rede neural de aprendizado de máquina. Uma
rede neural é um modelo de aprendizagem de máquina (machine learning) que consegue aprender padrões e fazer previsões
com base em
dados de entrada. Para treinar uma rede neural, é preciso fornecer a ela um conjunto de instâncias de treino.

### Definição de instâncias

Instância é um termo usado em aprendizagem de máquina (machine learning) para se referir a um exemplo ou caso de
treino. Em outras palavras, uma instância é um conjunto de dados usado para treinar um modelo de aprendizado
de máquina. Por exemplo, se estivermos a treinar um modelo de classificação de imagens, cada imagem seria uma instância
de treino.

### Aprendizagem supervisionada

Aprendizado por instância é um método de aprendizagem onde o modelo é treinado com base em um conjunto de
instâncias individuais.

Em vez de tentar generalizar a partir de uma abundância de dados, o modelo é treinado
para fazer previsões precisas para cada instância individualmente. Isso pode ser útil em casos em que o conjunto de
dados é muito pequeno ou quando cada instância é muito importante e precisa ser tratada de maneira muito específica.

No entanto, o aprendizado por instância também tem algumas desvantagens. Como o modelo é treinado com base em cada
instância individualmente, ele pode não ser tão bom em generalizar para novos conjuntos de dados. Além disso, o
treino pode ser mais demorado, pois, o modelo precisa processar cada instância individualmente.

### Formato de instâncias

As instâncias de treino são armazenadas num arquivo de texto com o formato CSV (comma-separated values). Cada
linha do arquivo representa uma instância e cada coluna representa um atributo.

Os primeiros 60 atributos representam os valores adquiridos pelo acelerômetro (x, y, z) numa taxa de 20Hz (20 amostras
por segundo). Os dois ultimos atributos representam o ID da atividade ('0' para 'downstairs, '1' para 'jogging', '2'
para 'sitting', '3' para 'standing', '4'
para 'upstairs', '5' para 'walking') e o ID do utilizador (1-36), respetivamente.

### Explicação da função 'create_instances'

A função 'create_instances' recebe como argumento os dados originais. Ela percorre todas as linhas do ficheiro '.CSV'
caso existam mais 20 linhas, e as mesmas pertençam ao mesmo utilizador e à mesma atividade, cria uma instância com os
valores dos 60 atributos e os dois ultimos atributos (ID da atividade e ID do utilizador). Caso contrário, passa para a
próxima linha.

Este processo é repetido até que não existam mais linhas no ficheiro '.CSV', e depois é guardado o resultado num
ficheiro '.CSV' com o nome 'instances.csv'.

## Passo 2: Criação dos 'folds' (com o algoritmo K-Fold)

### Introdução

A validação do modelo é uma etapa crucial na criação de um modelo de aprendizado de máquina. O objetivo é avaliar como o
modelo se comporta em dados que ele ainda não viu, para garantir que ele possa generalizar bem para novos dados. Uma das
técnicas mais comuns para validar um modelo é dividir os dados em dois conjuntos: um conjunto de treino e um
conjunto de teste. O modelo é treinado com o conjunto de treino e, em seguida, é avaliado com o conjunto de teste.

No entanto, às vezes é útil dividir o conjunto de treino em múltiplas "dobras" ou "folds", para ter uma ideia mais
precisa do desempenho do modelo. Isso é conhecido como validação cruzada.

Neste capítulo, discutiremos como criar folds de treino e teste usando o algoritmo K-fold validation. Explicaremos como
dividir os dados em folds de tamanho igual e como treinar e avaliar o modelo em cada uma dessas folds.

Também discutiremos como interpretar os resultados da validação cruzada e como usá-los para melhorar o desempenho do
modelo.

### Explicação da função 'create_k_fold_validation'

A função 'create_k_fold_validation' recebe como argumento o número de folds que se pretende criar.

Ela divide o conjunto de instâncias em 'k' partes iguais (onde em folds diferentes não existem instâncias com o mesmo ID
de utilizador), e para cada uma dessas partes, cria um conjunto de treino e um conjunto de teste. O conjunto de treino é
composto por todas as instâncias exceto as do fold atual, e o conjunto de teste é composto
pelas instâncias do fold atual. Este processo é repetido até que não existam mais folds.

## Passe 3: Normalização dos dados

### Introdução

A normalização dos dados é uma etapa importante na preparação dos dados para o treino de um modelo de aprendizado
de máquina. Ela consiste em transformar os dados de maneira a deixá-los com uma distribuição mais uniforme, o que pode
ajudar a melhorar o desempenho do modelo.

Na validação cruzada K-fold, a normalização dos folds é ainda mais importante, pois cada fold é usado como conjunto de
treino e teste em diferentes iterações. Se um fold tiver uma distribuição muito diferente dos outros folds, isso
pode afetar negativamente o desempenho do modelo naquela iteração.

Neste capítulo, discutiremos como foi feita a normalização dos folds resultantes do algoritmo K-fold validation e como
realizar essa normalização de maneira eficaz.

### Explicação da função 'normalizeData'

A função 'normalizeData' recebe como argumento o conjunto de instâncias (treino e teste) que se pretende normalizar e o
valor mínimo e máximo do conjunto de treino (obtido através da função getMinMax).

A função percorre todas as instâncias do conjunto de treino e altera o valor de cada atributo (x, y, z) para o valor
normalizado, calculado da seguinte forma:

(valor do atributo — valor mínimo do conjunto de treino) / (valor máximo do conjunto de treino — valor mínimo do
conjunto de treino)

Caso o valor máximo do conjunto de treino seja igual ao valor mínimo do conjunto de treino, o valor do atributo é
eliminado.

## Passo 4: Criação da Rede Neuronal

### Introdução e Explicação da Rede Neuronal

As redes neurais são um tipo poderoso de modelo de aprendizado de máquina, capaz de realizar tarefas complexas de
processamento de dados, como classificação, regressão e deteção de padrões. Elas são inspiradas na estrutura do cérebro
humano sendo compostas por camadas de neurónios interconectados que trabalham juntos para realizar tarefas específicas.

Neste capítulo, apresentarei os principais conceitos envolvidos na criação de uma rede neuronal (em python) e
discutiremos como implementar uma rede neural num trabalho prático. 