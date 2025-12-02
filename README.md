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


## Implementação Do Novo Bitonic Sort
O novo Bitonic Sort consegue ordenar vetores de tamanho arbitrário, não dependendo da
quantidade de threads no bloco, o que é possivel ao atribuir a cada thread a
responsabilidade por comparar um numero X de pares de elementos, com X sendo igual a
metade do tamanho do vetor pela quantidade de threads (asssumindo que a quantidade de
threads é menor que metade do tamanho do vetor). Para vetores que não são múltiplos
de 2, valores de padding são adicionados na shared memory e o algoritmo roda para o
tamanho do próximo múltiplo de 2 do vetor.
Devido a restrições com os kernels do último trabalho, o número máximo de segmentos é
igual ao númeor máximo de threads por bloco, i.t 1024 threads.

## Ambiente de teste
Os teste foram feitos utilizando a máquina Orval do Dinf, que roda uma GTX 750 Ti, usando
o maior numero de threads possivel na GPU para o blockBitonicSort, i.e 1024 threads
(média de 100 repetições).


## Resultados

### Tempos:
Elementos:      1M      2M      4M      8M
MPP (ms):       6.2     8.3     17.1    54.919
Thrust (ms):    2.4     4.5     8.8     17.287
Speedup:        0.39x   0.55x   0.51x   0.31x

### Vazão:
Elementos:               1M      2M      4M      8M
MPP (GElements/s):       0.16    0.241   0.233   0.146
Thrust (GElements/s):    0.406   0.437   0.454   0.463
