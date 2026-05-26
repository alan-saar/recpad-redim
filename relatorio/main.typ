#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Análise de Preservação Estrutural e Estabilidade em Redução de Dimensionalidade],
  abstract: [
    A análise de dados em alta dimensionalidade impõe severos desafios para tarefas de aprendizado de máquina e visualização, fenômeno amplamente conhecido como a maldição da dimensionalidade. Técnicas de redução de dimensionalidade visam mitigar esse problema projetando os dados em espaços de baixa dimensão enquanto tentam preservar propriedades topológicas ou geométricas originais. Este trabalho apresenta um estudo comparativo conceitual e experimental envolvendo três métodos proeminentes: o método linear clássico Principal Component Analysis (PCA) e os métodos não lineares estocásticos t-distributed Stochastic Neighbor Embedding (t-SNE) e Uniform Manifold Approximation and Projection (UMAP). Utilizando o conjunto de dados MNIST com amostragem estratificada, os algoritmos foram analisados qualitativa e quantitativamente sob a ótica de fidelidade de vizinhança (Trustworthiness), continuidade estrutural (Continuity), densidade dos agrupamentos (Silhouette Score) e tempo computacional. Adicionalmente, investigou-se a estabilidade e robustez dos métodos não lineares frente a 10 sementes aleatórias distintas e variações críticas de hiperparâmetros (perplexity e n_neighbors). Por fim, realizou-se uma avaliação downstream por meio de classificação k-NN para mensurar a utilidade analítica das projeções bidimensionais geradas.
  ],
  authors: (
    (
      name: "Alan Saar",
      department: [PPCOMP],
      organization: [Instituto Federal do Espírito Santo (IFES)],
      location: [Serra, ES - Brasil],
      email: "alan.saar@ifes.edu.br",
    ),
    (
      name: "Leonardo Herkenhoff",
      department: [PPCOMP],
      organization: [Instituto Federal do Espírito Santo (IFES)],
      location: [Serra, ES - Brasil],
      email: "leonardo.herkenhoff@ifes.edu.br",
    ),
  ),
  index-terms: ("Redução de Dimensionalidade", "PCA", "t-SNE", "UMAP", "Fidelidade Estrutural", "MNIST"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introdução
Com o avanço da capacidade computacional e a coleta massiva de informações, o processamento de dados contendo dezenas, centenas ou milhares de atributos tornou-se ubíquo na ciência da computação e em áreas correlatas @lecun1998. No entanto, trabalhar em espaços de alta dimensionalidade introduz a chamada "maldição da dimensionalidade". À medida que o número de dimensões cresce, o volume do espaço geométrico aumenta exponencialmente, fazendo com que os dados disponíveis se tornem extremamente esparsos. Sob essa condição, métricas de distância tradicionais, como a distância euclidiana, perdem poder discriminativo, pois a diferença entre a distância do ponto mais próximo e a do ponto mais distante tende a zero.

Técnicas de redução de dimensionalidade desempenham um papel vital para contornar esses limites, compactando as informações essenciais em representações de baixa dimensão (geralmente duas ou três dimensões para fins de visualização). Essa compactação busca encontrar redundâncias e correlações intrínsecas nos dados originais.

Este relatório apresenta uma análise comparativa profunda envolvendo três métodos obrigatórios: PCA @pearson1901, t-SNE @vandermaaten2008 e UMAP @mcinnes2018. O estudo foca em responder a questões críticas relacionadas à preservação de relações geométricas locais e globais, à estabilidade estocástica e reprodutibilidade científica das projeções, à sensibilidade extrema a hiperparâmetros e à capacidade de os embeddings reterem informações semânticas úteis para tarefas subsequentes (downstream) de reconhecimento de padrões.

= Fundamentação Teórica
A redução de dimensionalidade pode ser abordada por vias lineares ou não lineares. A escolha de cada paradigma impõe diferentes compromissos (*trade-offs*) entre velocidade computacional e fidelidade estrutural.

== Principal Component Analysis (PCA)
O PCA é um método clássico linear @pearson1901 que busca projetar os dados em um novo sistema de coordenadas ortogonais, chamados de componentes principais. Esses componentes são ordenados pela quantidade de variância dos dados que capturam. Matematicamente, a primeira direção de projeção $w_1$ é definida por:
$ w_1 = "argmax"_(||w||=1) (||X w||^2) $
O PCA é determinístico, extremamente rápido e excelente na preservação de estruturas geométricas de escala macroscópica (globais). Contudo, por realizar apenas transformações lineares (rotações e escalonamentos), falha em desenrolar subespaços curvos de baixa dimensão imersos em alta dimensionalidade (*manifolds*).

== t-SNE
O t-SNE @vandermaaten2008 é um algoritmo não linear estocástico voltado à visualização. Ele converte distâncias euclidianas entre pontos no espaço de alta dimensão em probabilidades condicionais $p_(j|i)$ que representam similaridades de vizinhança:
$ p_(j|i) = exp(-||x_i - x_j||^2 / (2 sigma_i^2)) / (sum_(k != i) exp(-||x_i - x_k||^2 / (2 sigma_i^2))) $
No espaço de baixa dimensão, a probabilidade conjunta $q_(i j)$ é modelada usando uma distribuição t-Student com um grau de liberdade (equivalente à distribuição de Cauchy) para atenuar o *crowding problem* (fenômeno onde a área de baixa dimensão é insuficiente para acomodar as distâncias de alta dimensão, empurrando clusters para cima uns dos outros):
$ q_(i j) = (1 + ||y_i - y_j||^2)^(-1) / (sum_(k != l) (1 + ||y_k - y_l||^2)^(-1)) $
O algoritmo minimiza a divergência de Kullback-Leibler (KL) entre as distribuições $P$ e $Q$ via gradiente descendente estocástico. Embora excelente na preservação local, o t-SNE é sensível à inicialização aleatória, lento e apresenta limitações na preservação de relações globais.

== UMAP
O UMAP @mcinnes2018 baseia-se em conceitos matemáticos de geometria riemanniana e topologia algébrica. Ele assume que os dados estão distribuídos ao longo de um manifold localmente conexo. Diferente do t-SNE, o UMAP modela as similaridades no espaço original e projetado por meio de grafos fuzzy conexos e otimiza a representação minimizando a entropia cruzada difusa, em vez da divergência KL. Isso permite ao UMAP preservar melhor a macroestrutura global, além de apresentar eficiência computacional significativamente superior ao t-SNE.

== Métricas de Avaliação Estrutural
Para ir além da mera análise visual superficial, duas métricas clássicas de fidelidade topológica introduzidas por Venna e Kaski @venna2001 @venna2006 são utilizadas:

1. *Trustworthiness (Confiabilidade)*: Avalia a ocorrência de falsos vizinhos na projeção. Mede se pontos próximos no espaço bidimensional eram de fato próximos no espaço original de alta dimensão:
$ T(k) = 1 - 2 / (N k (2N - 3k - 1)) sum_(i=1)^N sum_(j in U_i^(k)) (r(i, j) - k) $ <eq:trust>
Onde $U_i^(k)$ representa os pontos que estão entre os $k$-vizinhos mais próximos no espaço projetado, mas não no original, e $r(i, j)$ é o rank de distância do ponto $j$ em relação a $i$ no espaço de alta dimensão.

2. *Continuity (Continuidade)*: Mede se vizinhos próximos no espaço original foram separados ou "quebrados" na projeção (falsas extrusões). Matematicamente, a *Continuity* é o dual elegante da *Trustworthiness*, calculada invertendo-se os papéis dos dados de alta e baixa dimensão:
$ C(k) = "trustworthiness"("Espaço Projetado", "Espaço Original") $

3. *Silhouette Score*: Mede a qualidade do agrupamento semântico baseado nas classes reais:
$ s_i = (b_i - a_i) / (max(a_i, b_i)) $ <eq:sil>
onde $a_i$ é a distância média intracluster e $b_i$ é a distância média para o cluster vizinho mais próximo @rousseeuw1987.

= Metodologia
Os experimentos foram executados utilizando uma subamostra estratificada de $N = 3000$ imagens do dataset MNIST @lecun1998, garantindo representatividade exata de todos os dígitos (0 a 9). Os dados foram normalizados dividindo os valores dos pixels por 255.0. 

Para a execução do t-SNE, aplicou-se uma redução preliminar via PCA para 50 componentes principais para redução de ruído e aceleração computacional, prática padrão sugerida por van der Maaten @vandermaaten2008. Os experimentos foram desenvolvidos em linguagem Python utilizando as bibliotecas `scikit-learn` e `umap-learn`. As imagens geradas foram salvas diretamente para composição automática deste artigo.

= Resultados e Discussão

== Parte A — Fidelidade Estrutural
As projeções bidimensionais geradas para os três métodos avaliados estão expostas nas figuras a seguir.

#figure(
  image("img/projection_pca.png", width: 78%),
  caption: [Projeção bidimensional obtida por meio do método linear PCA.],
) <fig:pca>

#figure(
  image("img/projection_tsne.png", width: 78%),
  caption: [Projeção bidimensional obtida por meio do método não linear t-SNE.],
) <fig:tsne>

#figure(
  image("img/projection_umap.png", width: 78%),
  caption: [Projeção bidimensional obtida por meio do método não linear UMAP.],
) <fig:umap>

A análise visual revela comportamentos distintos. O PCA (@fig:pca) gera uma projeção contínua e sobreposta no centro, incapaz de segregar as classes com clareza. Isso ocorre porque transformações lineares não conseguem capturar as fronteiras de decisão complexas e não lineares das imagens.

O t-SNE (@fig:tsne) e o UMAP (@fig:umap) conseguem desenrolar as estruturas e criar ilhas/agrupamentos isolados para quase todas as classes de dígitos. O t-SNE foca intensamente na separação local das classes, deixando os clusters dispersos. O UMAP demonstra visualmente maior coesão de vizinhança global, posicionando clusters conceitualmente parecidos mais próximos no mapa bidimensional.

Os resultados quantitativos calculados para os três métodos ($k=15$ vizinhos para Trustworthiness e Continuity) estão consolidados na @tab:structural.

#figure(
  caption: [Métricas quantitativas de fidelidade estrutural e desempenho temporal dos três métodos de projeção.],
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    inset: 7pt,
    stroke: (x, y) => if y <= 1 { (bottom: 0.5pt, top: 0.5pt) } else if y == 4 { (bottom: 0.5pt) },
    table.header([*Método*], [*Trustworthiness*], [*Continuity*], [*Silhouette Score*], [*Tempo (s)*]),
    [PCA], [0.9575], [0.8122], [0.0381], [0.0392],
    [t-SNE], [0.9859], [0.9634], [0.3541], [10.4281],
    [UMAP], [0.9904], [0.9427], [0.3804], [4.1127],
  ),
) <tab:structural>

Como observado na @tab:structural, o PCA possui o menor tempo computacional, porém seu Silhouette Score é extremamente baixo ($0.0381$), refletindo a enorme sobreposição das classes. O t-SNE obteve excelente fidelidade de vizinhança local, com destaque para a continuidade estrutural ($0.9634$), evitando quebras de estruturas originais.

O UMAP apresentou a melhor métrica de Confiabilidade (*Trustworthiness* = $0.9904$), garantindo que pontos próximos na projeção são verdadeiramente vizinhos originais. Adicionalmente, o UMAP obteve o maior Silhouette Score ($0.3804$) e uma eficiência computacional extraordinária, rodando mais de duas vezes mais rápido que o t-SNE.

== Parte B — Estabilidade e Robustez

=== Estabilidade das Sementes Aleatórias
Por utilizarem inicializações estocásticas, o t-SNE e o UMAP foram executados sob 10 sementes aleatórias distintas para mensurar a reprodutibilidade estrutural e a variabilidade das projeções. Os resultados estatísticos de dispersão de desempenho estão sumarizados no gráfico de caixas da @fig:stability.

#figure(
  image("img/stability_boxplot.png", width: 85%),
  caption: [Diagrama de caixa (Boxplot) comparando a estabilidade de Trustworthiness, Continuity e Silhouette Score sob 10 sementes distintas para t-SNE e UMAP.],
) <fig:stability>

A análise da @fig:stability demonstra que ambos os métodos possuem altíssima estabilidade nas métricas quantitativas. A variação (desvio padrão) é extremamente baixa, sugerindo que embora a disposição visual geográfica dos clusters possa sofrer rotações e espelhamentos físicos devido à aleatoriedade das sementes, as propriedades topológicas fundamentais e a qualidade de vizinhança são mantidas de forma muito robusta.

=== Sensibilidade a Hiperparâmetros
A escolha de hiperparâmetros influencia de forma drástica a geometria final dos embeddings. A @fig:tsnehyp ilustra a varredura da perplexidade no t-SNE, enquanto a @fig:umaphyp apresenta o impacto do número de vizinhos (`n_neighbors`) no UMAP.

#figure(
  image("img/tsne_hyperparameter_sweep.png", width: 90%),
  caption: [Impacto da perplexidade nas projeções bidimensionais do t-SNE. Valores baixos focam em estruturas finas e locais; valores altos buscam preservar a globalidade a custo de coesão local.],
) <fig:tsnehyp>

#figure(
  image("img/umap_hyperparameter_sweep.png", width: 90%),
  caption: [Impacto do parâmetro n_neighbors nas projeções do UMAP. Controla o balanço entre fidelidade estrutural local e global.],
) <fig:umaphyp>

Para o t-SNE (@fig:tsnehyp), perplexidade muito baixa ($5$) fragmenta as classes de dígitos em dezenas de pequenos agrupamentos artificiais sem coerência semântica. A perplexidade intermediária ($30$) revela-se ideal para segregar os dígitos em suas classes reais. No UMAP (@fig:umaphyp), o aumento do número de vizinhos de $5$ para $80$ aglutina progressivamente as projeções, fundindo clusters locais e forçando a priorização do arranjo global.

== Parte C — Avaliação Downstream (k-NN)
Para avaliar quantitativamente a utilidade analítica das projeções na retenção de informações discriminativas semânticas, treinou-se um classificador k-NN com $k=5$ vizinhos usando validação particionada (80% treino, 20% teste). Os desempenhos de acurácia no espaço original e nos espaços projetados 2D estão sintetizados no gráfico de barras da @fig:downstream.

#figure(
  image("img/knn_downstream_bar.png", width: 85%),
  caption: [Comparativo de Acurácia de Classificação k-NN (k=5) utilizando o espaço original 784D contra as representações 2D projetadas.],
) <fig:downstream>

A classificação no espaço original de 784 dimensões alcança excelente acurácia ($0.9417$). A projeção do PCA em 2D degrada massivamente a capacidade do classificador, obtendo apenas $0.4167$ de acurácia.

Em contrapartida, as representações bidimensionais do t-SNE ($0.9000$) e do UMAP ($0.9083$) retêm quase toda a informação preditiva original. O k-NN obtém acurácias muito próximas à do espaço de alta dimensão original, demonstrando de forma empírica extraordinária preservação de vizinhança semântica mesmo comprimindo o espaço original em $99.7\%$ (de 784 para 2 dimensões).

= Discussão Crítica
Embora o t-SNE e o UMAP gerem projeções visualmente atraentes e clusters bem definidos, a visualização bidimensional de dados de alta dimensionalidade exige cautela metodológica para evitar conclusões precipitadas (*overinterpretation*):

1. *Distâncias Inter-clusters*: No t-SNE, a distância física entre ilhas separadas não possui significado geométrico real. Duas ilhas distantes no gráfico bidimensional não necessariamente representam classes muito distintas em alta dimensão. O UMAP suaviza essa limitação, apresentando maior coerência global, mas ainda não se deve inferir relações métricas rígidas a partir da distância visual.
2. *Fenômeno de Crowding*: A perda de volume intrínseca à redução para duas dimensões inevitavelmente força distorções topológicas. A projeção bidimensional deve ser vista prioritariamente como uma ferramenta qualitativa de exploração rápida de dados, exigindo validação quantitativa complementar por meio de métricas como as apresentadas na @tab:structural.

= Conclusões
Este estudo permitiu compreender de forma prática e teórica os fundamentos da redução de dimensionalidade. O método linear PCA provou-se inadequado para separação visual e classificação de dados com comportamento complexo não linear como o MNIST, embora sirva como um excelente pré-processamento de aceleração.

Entre os métodos não lineares, o t-SNE destaca-se pela preservação da continuidade estrutural local, sendo ideal para explorar minúcias e vizinhanças imediatas. O UMAP, por sua vez, demonstrou ser o método mais equilibrado e poderoso para tarefas científicas robustas no MNIST. Ele alcançou a maior fidelidade quantitativa (Trustworthiness), melhor separação global dos clusters, maior desempenho de classificação downstream (k-NN) e custos computacionais de processamento drasticamente reduzidos.

