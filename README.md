# CUDA-MPPSort

## Objetivos
Implementar um algoritmo eficiente de ordenação paralela usando particionamento múltiplo
Medir o tempo e vazão do algoritmo completo, bem como dos diversos kernels utilizados para
obter o vetor ordenado, usando o algorimo de ordenação da biblioteca thrust como referência.

## Funcionamento
O algoritmo funciona ao dividir o vetor em segmentos (ou partições) de amplitudes iguais,
mapeando as chaves do vetor aos diversos segmentos, ordenando as partições globalmente, e em seguida
ordenando os elementos internos de cada segmento paralelamente, ordenando multiplas partições simultaneamente.

### Detalhes
O algoritmo requer a implementação de 5 kernels, que cumprem as seguintes funções, de forma resumida:

1) Constroi um histograma que armazena a quantidade de elementos que cada partição irá comportar. Também obtém uma
matriz que mapeia a quantidade de elementos em cada partição que cada bloco haverá de ordenar na fase de partição.

2) Realiza operação de redução (Scan) no histograma obtido pelo kernel 1, a fim de obter os indices correspondentes ao 
começo de cada partição

3) Realiza o scan vertical da matriz do mapa de blocos - partições, obtendo o indice interno a cada partição no qual cada
bloco havera de colocar os seus elementos

4) Utiliza das estruturas previamente obtidas para organizar as partições de forma ordenada no vetor. Depois desse algoritmo, todas
as partições tem seus elementos internos juntos, com as partições ordenadas em sequência, mas os elementos internos dos segmentos
ainda estão desordenados

5) Percorre os segmentos de forma paralela, ordenando internamente NBLOCOS particões por passo


## Implementação
Infelizmente, entradas muito altas ou número de partições grandes demais geram bugs e acabam ordenando o vetor incorretamente.
Quando partições são muito grandes e estouram o limite de memória compartilhada dos blocos, o thrust sort é chamado para ordená-los,
adicionando um overhead significativo que faz com que a implementação perca para o algoritmo de referência.
Os passos da ordenação ainda são realizados mesmo com um reultado incorreto e, tanto essas saídas incorretas quanto as entradas 
menores que funcionam corretamente conseguem bater o tempo do algoritmo thrust, desde que o tamanho das partições não seja
maior que o limite da memória compartilhada.
Junto do trabalho está um gráfico mostrando o resultado de um teste com 10000 elementos e 16 partições, executado com 100
repetições.

## Ambiente de teste
Os teste foram feitos utilizando a máquina Orval do Dinf, que roda uma GTX 750 Ti


## Resultados
======== Kernel Timing (Average over 100 runs) ========
Kernel 1 (blockAndGlobalHistogram): 0.015 s
Kernel 2 (globalHistogramScan):     0.009 s
Kernel 3 (verticalScanHH):          0.009 s
Kernel 4 (partitionKernel):         0.015 s
Kernel 5 (bitonicSort + merge):    0.083 s
Total mppSort time:                 0.131 s

mppSort Throughput: 0.076 GElements/s
Thrust sort time:   0.201 ms
Thrust Throughput:  0.050 GElements/s


