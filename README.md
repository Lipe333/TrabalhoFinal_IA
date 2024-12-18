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


### Escopo:

É dada uma instância de 100 trens de Michalsky, gerada aleatoriamente, contendo diversos atributos como: quantidade de carros (locomotiva e vagões); formato e comprimento dos vagões; formato e quantidade de cargas; a quantidade de eixo com rodas; etc. Os dados foram apresentados em um arquivo csv de nome: trains-uptated.csv.

Propõe-se que os trens teriam seus destinos considerados para EAST ou WEST conforme sua apresentação na lista, convencionando que os trens da esquerda (1-25 e 51-75) iriam para EAST e os da direita (26-50 e 76-100) iriam para WEST.

A ideia do estudo seria, primeiro, identificar, por análise, as características que poderiam evidenciar padrões de determinação da direção dos trens, usando para tanto algoritmos de clustering e um programa auxiliar fornecido capaz de apontar similaridades entre atributos, o split_dataset.py. Dessa análise, o objetivo era extrair ou gerar supostos axiomas que serviriam de base para a criação de um modelo LTN que possa testar as regras para classificação dos trens no dataset, com a implementação de uma solucão em LTNTorch.
