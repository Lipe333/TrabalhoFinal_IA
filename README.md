# TrabalhoFinal_IA
Universidade Federal do Amazonas  
Instituto de Computação - Icomp  
Disciplina: Inteligência Artificial - Período: 2024/2  
Trabalho Prático III  

## Equipe:

Caio Antunes (caio.antunes@icomp.ufam.edu.br)  
Felipe Spitale (felipe.spitale@icomp.ufam.edu.br)   
Marcello Cipriano (marcello.cipriano@icomp.ufam.edu.br)   
Pedro Henrique (pedro.santos@icomp.ufam.edu.br)   

### Observação:

O data set usado foi o arquivo `trains-uptated.csv`, mas ele foi processado para poder ser usado no treinamento.
O resultado do data set processado é o arquivo `processedDataset.csv`

### Escopo:

É dada uma instância de 100 trens de Michalsky, gerada aleatoriamente, contendo diversos atributos como: quantidade de carros (locomotiva e vagões); formato e comprimento dos vagões; formato e quantidade de cargas; a quantidade de eixo com rodas; etc. Os dados foram apresentados em um arquivo csv de nome: trains-uptated.csv.

Propõe-se que os trens teriam seus destinos considerados para EAST ou WEST conforme sua apresentação na lista, convencionando que os trens da esquerda (1-25 e 51-75) iriam para EAST e os da direita (26-50 e 76-100) iriam para WEST.

A ideia do estudo seria, primeiro, identificar, por análise, as características que poderiam evidenciar padrões de determinação da direção dos trens, usando para tanto algoritmos de clustering e um programa auxiliar fornecido capaz de apontar similaridades entre atributos, o split_dataset.py. Dessa análise, o objetivo era extrair ou gerar supostos axiomas que serviriam de base para a criação de um modelo LTN que possa testar as regras para classificação dos trens no dataset, com a implementação de uma solucão em LTNTorch.


## Questão 1:
### A)  

### Agrupar trens por similaridade usando algoritmo de clustering:  

Data columns (total 33 columns):  
    Column                       Non-Null Count  Dtype    
  
 0   Number_of_cars               100 non-null    float64  
 1   Number_of_different_loads    100 non-null    float64  
 2   num_wheels1                  100 non-null    float64  
 3   length1                      100 non-null    float64  
 4   shape1                       100 non-null    float64  
 5   num_loads1                   100 non-null    float64  
 6   load_shape1                  100 non-null    float64  
 7   num_wheels2                  100 non-null    float64  
 8   length2                      100 non-null    float64  
 9   shape2                       100 non-null    float64  
 10  num_loads2                   100 non-null    float64  
 11  load_shape2                  100 non-null    float64  
 12  num_wheels3                  100 non-null    float64  
 13  length3                      100 non-null    float64  
 14  shape3                       100 non-null    float64  
 15  num_loads3                   100 non-null    float64  
 16  load_shape3                  100 non-null    float64  
 17  num_wheels4                  100 non-null    float64  
 18  length4                      100 non-null    float64  
 19  shape4                       100 non-null    float64  
 20  num_loads4                   100 non-null    float64  
 21  load_shape4                  100 non-null    float64  
 22  Rectangle_next_to_rectangle  100 non-null    float64  
 23  Rectangle_next_to_triangle   100 non-null    float64  
 24  Rectangle_next_to_hexagon    100 non-null    float64  
 25  Rectangle_next_to_circle     100 non-null    float64  
 26  Triangle_next_to_triangle    100 non-null    float64  
 27  Triangle_next_to_hexagon     100 non-null    float64  
 28  Triangle_next_to_circle      100 non-null    float64  
 29  Hexagon_next_to_hexagon      100 non-null    float64  
 30  Hexagon_next_to_circle       100 non-null    float64  
 31  Circle_next_to_circle        100 non-null    float64  
 32  Class_attribute              100 non-null    float64  

### Os Trens são visualizados de forma interativa no espaço 2D usando Plotly, cada ponto representa um trem.

![newplot](https://github.com/user-attachments/assets/c2adc2f8-36d4-483e-9c66-3b1d5db3b554)

#### B)  
Buscar similaridades por análise e gerar supostos axiomas:  
Separação dos clusters em dois dataframes para auxiliar análise de características e extração dos axiomas.  

Cluster 0: [(4, 1), (7, 1), (9, 1), (11, 1), (12, 1), (14, 1), (17, 1), (18, 1), (24, 1), (25, 1), (26, 0), (30, 0), (31, 0), (33, 0), (34, 0), (36, 0), (44, 0), (46, 0), (50, 0), (52, 1), (56, 1), (59, 1), (60, 1), (61, 1), (63, 1), (66, 1), (68, 1), (71, 1), (74, 1), (78, 0), (79, 0), (80, 0), (82, 0), (85, 0), (87, 0), (93, 0), (94, 0), (96, 0)]  

Cluster 1: [(1, 1), (2, 1), (3, 1), (5, 1), (6, 1), (8, 1), (10, 1), (13, 1), (15, 1), (16, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (27, 0), (28, 0), (29, 0), (32, 0), (35, 0), (37, 0), (38, 0), (39, 0), (40, 0), (41, 0), (42, 0), (43, 0), (45, 0), (47, 0), (48, 0), (49, 0), (51, 1), (53, 1), (54, 1), (55, 1), (57, 1), (58, 1), (62, 1), (64, 1), (65, 1), (67, 1), (69, 1), (70, 1), (72, 1), (73, 1), (75, 1), (76, 0), (77, 0), (81, 0), (83, 0), (84, 0), (86, 0), (88, 0), (89, 0), (90, 0), (91, 0), (92, 0), (95, 0), (97, 0), (98, 0), (99, 0), (100, 0)]  

Cluster 0 - Quantidade de trens para EAST: 20, e para WEST: 18  
Cluster 1 - Quantidade de trens para EAST: 30, e para WEST: 32  

Com a aplicação do algoritmo de clustering, foi possível conferir nitidamente a formação de um agrupamento (Cluster 0) contendo apenas os trens com 5 carros (1 locomotiva e 4 vagões) e um outro agrupamento (Cluster 1) contendo os demais exemplos de trens com menos de 5 carros.

Ocorre que essa divisão ainda mantém muita aproximada de 50% a proporção de trens que se destinam a cada lado (EAST e WEST) em ambos os clusters, de modo que exige a adoção de outros critérios para análise e identificação de potenciais regras de determinação do direcionamento aplicáveis a essa instância de Michalsky.

Para essa tarefa complementar, definimos abaixo então a função split_dataset, a partir do código em python fornecido na especificação do trabalho, com algumas adaptações para extrair dados relevantes para a síntese dos axiomas com base na análise de padrões dos clusters, assim como da totalidade da amostra.

Após a inclusão dessas novas colunas, aplicamos novamente a função split_dataset sobre a tabela, mas tão somente considerando a relevância das 4 colunas destacadas nos apontamentos acima:

a coluna 'Number_of_cars' percebida com aplicação do algoritmo de clustering;
e as três novas colunas criadas: 'quant_car_long'; 'has_load_hexagon_shape'; 'has_num_loads_car_above1'.


Dessa forma, restaram mais evidentes os pontos de similaridades investigados, tornando possível apresentar supostos axiomas para determinar a orientação dos trens com maior potencial de precisão.

Antes porém de expressar os identificados axiomas, deixemos clara a relação de predicados do problema já oferecidos na especificação do trabalho, com o acréscimo de alguns decorrentes da criação dos novos dados expressos na tabela de características encontradas:

Predicados dados:

 - num_cars(t, nc), em que t ∊ [1..100] e nc ∊ [3..5] (nc denota o número de carros do trem incluindo a locomotiva).  
 - num_loads(t, nl) em que t ∊ [1..100] e nl ∊ [1..4].  
 - num_wheels(t, c, w) em que t ∊ [1..100] e c ∊ [1..4] e w ∊ [2..3].  
 - length(t, c, l) em que t ∊ [1..100] e c ∊ [1..4] e l ∊ [1..2] (1 denota curto e 2 longo)  
 - shape(t, c, s) em que t ∊ [1..100] e c ∊ [1..4] e s ∊ [1..15] (um número para cada forma).  
 - num_cars_loads(t, c, ncl) em que t ∊ [1..100] e c ∊ [1..4] e ncl ∊ [0..3].  
 - load_shape(t, c, ls) em que t ∊ [1..100] e c ∊ [1..4] e ls ∊ [1..4].   
 - next_crc(t, c, x) em que t ∊ [1..100] e c ∊ [1..4] e x ∊ [0..1], em que o vagão c do trem t tem um vagão adjacente com cargas em círculo.  
 - next_hex(t, c, x) em que t ∊ [1..100] e c ∊ [1..4] e x ∊ [0..1], em que o vagão c do trem t tem um vagão adjacente com cargas em hexágono.  
 - next_rec(t, c, x) em que t ∊ [1..100] e c ∊ [1..4] e x ∊ [0..1], em que o vagão c do trem t tem um vagão adjacente com cargas em retângulo.  
 - next_tri(t, c, x) em que t ∊ [1..100] e c ∊ [1..4] e x ∊ [0..1], em que o vagão c do trem t tem um vagão adjacente com cargas em triângulo.  
Predicados novos identificados:  

 - num_cars_long(t, nclg), em que t ∊ [1..100] e nclg ∊ [0..4], indicando a quantidade de vagões de comprimento longo.  
 - has_num_loads_car_above1(t, nlab1), em que t ∊ [1..100] e nlab1 ∊ [0..1], indicando se o trem tem algum vagão com uma carga superior a uma unidade.  
Obs.: Como predicado relativo à nova coluna 'has_load_hexagon_shape' é possível utilizar o expresso no número 7 acima.  

## Questão 2:
#### Definindo modelo - o modelo será uma rede neural para prever a classe do trem (leste ou oeste):
### Gráfico de Acurácia
![Sem título](https://github.com/user-attachments/assets/09af2d48-5c32-4d77-84e2-1cfadc0a5454)
### Gráfico de Perda
![Sem título](https://github.com/user-attachments/assets/9f695ede-a7f2-4d21-80b1-76630b338d72)

