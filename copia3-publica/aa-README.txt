CI1008 2sem25

Este diretorio contem o esqueleto do programa
  segmented-sort-bitonic.cu
e as 3 versoes em binario que rodam na nv00
(com o codigo do prof para voce tentar fazer o seu e
saber se está dentro do esperado, ou pelo menos proximo!)

os binarios executaveis foram produzidos para a nv00
  segmented-sort-bitonic-256
  segmented-sort-bitonic-512
  segmented-sort-bitonic-1024
  
cada binario foi compilado com um numero
de THREADS_PER_BLOCK indicado
para voce saber qual configuracao serah melhor  
assim,
voce pode fazer a mesma coisa com o seu kernel
(i.e. gerar 3 binarios e testar o melhor
      para cada situacao de segmentos)

tabem contem um shell de compilacao
que compila em varias maquinas
voce pode mudar para incluir a
compilacao na sua maquina tambem

MAS
ao final, voce vai reportar seus
resultados rodando na nv00

--W

