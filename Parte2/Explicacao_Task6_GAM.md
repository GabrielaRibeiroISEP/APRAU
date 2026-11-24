# Tarefa 6 (GAM) explicada passo a passo

Este ficheiro explica, como se fosse a primeira vez que ouves falar do assunto, o que está no notebook `APRAU/Parte2/task6_GAM.ipynb` e o que foi alterado em 24 de novembro. Não é preciso saber nada de antemão. Os termos novos são explicados quando aparecem e há um glossário no fim.

---

## 1. O problema, em poucas frases

O dataset tem 2000 doentes. Cada um tem dezenas de variáveis (42 depois de codificar as categóricas; por exemplo idade, tensão arterial, `HbA1c_level`, se fuma, etc.) e uma etiqueta com a classe: `N`, `HY` ou `DI`. O objetivo é construir um modelo que, olhando para as colunas de um doente, adivinha a classe.

O enunciado pede uma estratégia chamada **One-vs-Rest** (uma contra as restantes). Em vez de um modelo que escolhe entre três classes, constroem-se três modelos de sim ou não: "é `N` ou não?", "é `HY` ou não?", "é `DI` ou não?". Depois compara-se qual dos três funciona melhor, escolhe-se esse e estuda-se a fundo.

O enunciado pede três coisas:

1. Construir os três modelos, compará-los e escolher o melhor, justificando.
2. No escolhido, ver como cada variável influencia a resposta (a influência é uma reta ou uma curva? isso bate certo com a análise exploratória da tarefa 2?).
3. Comparar o GAM com uma regressão logística normal: ter curvas em vez de retas melhora alguma coisa?

---

## 2. Ideias de base

### Treino, validação e teste

Imagina que estás a estudar para um exame. Os exercícios que fazes em casa são o **treino**. O exame a sério é o **teste**, e só o fazes no fim, uma vez. Se já tivesses visto as perguntas do exame, a nota não diria nada sobre o que sabes.

Por isso os 2000 doentes foram divididos em 1600 para treino (80%) e 400 para teste (20%). O teste só é usado no fim para ver como o modelo se porta com doentes que nunca viu.

A **validação cruzada** é um simulacro de exame feito só com os 1600 do treino. Dividem-se em 5 partes; treina-se com 4 e testa-se na quinta, e repete-se 5 vezes para que cada parte seja a "quinta" uma vez. O notebook repete tudo isto 3 vezes com divisões diferentes, o que dá 15 resultados por modelo. Assim tens uma média e uma ideia da variação, e não só um número solto.

### Regressão logística e GAM

A **regressão logística** é o modelo clássico. Para cada variável dá um único número (um coeficiente) que diz "quanto mais alta esta variável, mais (ou menos) provável a classe". O efeito é sempre uma **reta**: se subir de 10 para 20 tem o mesmo efeito que de 50 para 60.

Um **GAM** (Generalized Additive Model) mantém a ideia de somar o efeito de cada variável, mas deixa cada efeito ser uma **curva**. Por exemplo, o álcool pode não influenciar nada a meio e influenciar muito nos extremos. Se todas as curvas forem retas, o GAM é igual à regressão logística. Por isso a pergunta do enunciado ("a flexibilidade melhora?") é a pergunta certa.

Na biblioteca `pygam` há três tipos de termo, que o notebook escolhe consoante a variável:

- `s()` é uma curva livre, para variáveis contínuas (idade, `HbA1c_level`);
- `f()` dá um efeito próprio a cada nível, para variáveis com poucos níveis ordenados (por exemplo `bmi_level`, com 4 níveis);
- `l()` é uma reta, para variáveis sim ou não (`gender`, `family_history`), onde uma curva não faz sentido.

### O parâmetro λ (lambda)

Uma curva muito flexível pode serpentear para passar por todos os pontos, incluindo o ruído. O **λ** é o travão: quanto maior, mais a curva é obrigada a ser suave (no limite, uma reta); quanto menor, mais livre. O modelo escolhe o λ sozinho numa pequena lista de valores (`gridsearch`).

### Graus de liberdade efetivos (EDoF)

É uma medida de "quanta curva" o modelo está a usar. Uma reta vale 1, e uma curva mais ondulada vale mais. O EDoF total do modelo dá a complexidade dele.

### Lasso

O Lasso é um método da tarefa 5 que serve para escolher variáveis: ajusta uma regressão logística e obriga os coeficientes das variáveis pouco úteis a ficarem exatamente em zero. As que sobram são as "mais relevantes". O enunciado manda usar estas.

### Métricas (como se mede se o modelo é bom)

- **ROC-AUC**: imagina que escolhes um doente `DI` e um que não é `DI` ao acaso. O ROC-AUC é a probabilidade de o modelo dar uma probabilidade mais alta ao `DI`. 0,5 é lançar uma moeda; 1 é perfeito.
- **PR-AUC**: parecido, mas mais exigente quando a classe é rara.
- **Recall**: dos `DI` verdadeiros, quantos o modelo apanha.
- **Precision**: dos que o modelo diz serem `DI`, quantos são mesmo.
- **F1**: junta recall e precision num só número.
- **Balanced accuracy**: a média do acerto em cada uma das duas classes (assim uma classe rara não é esquecida).
- **Log-loss**: mede se as probabilidades são boas, e não só se a decisão final está certa. Dizer "99%" e errar custa muito mais do que dizer "60%" e errar. **Aqui menor é melhor.**

### Sobreajuste (overfitting)

É quando o modelo "decora" o treino, incluindo o ruído, em vez de aprender o que é geral. Nota excelente nos exercícios de casa, nota fraca no exame. O sinal típico é uma **grande diferença entre o desempenho no treino e fora do treino**.

---

## 3. O que o notebook faz, por secções

**Secção 6.0, os preditores.** Repete-se o Lasso da tarefa 5 e ficam as variáveis que ele mantém. Duas delas (`age_normalized` e `age_level_Young`) são a mesma idade escrita de outra forma, e são retiradas, porque o GAM já trata a idade com uma curva livre. Ficam 16 preditores (6 contínuos, 3 ordinais e 7 binários).

**Secção 6.1, os três modelos.** Treinam-se os três modelos (`N`, `HY` e `DI` contra o resto), cada um em GAM e em regressão logística, com a validação cruzada. A tabela e os gráficos mostram as métricas. A hipótese escolhida é a de maior ROC-AUC, porque essa métrica não depende de a classe ser rara ou frequente, e por isso é justa para comparar as três.

**Secção 6.2, os efeitos no modelo escolhido.** Para cada variável desenha-se a curva estimada (o "efeito parcial"). Para saber se uma curva é mesmo necessária, compara-se, em cada variável, o modelo completo com outro em que só essa curva passa a reta. Se a reta fizer pior de forma consistente, a curva era necessária. Compara-se também a ordem das variáveis com a da análise exploratória da tarefa 2.

**Secção 6.3, GAM contra regressão logística.** Mesmas variáveis, mesmas divisões da validação cruzada, mais uma comparação no conjunto de teste. Inclui-se um terceiro modelo, o "GAM linear" (um GAM em que tudo é reta), para separar duas coisas: o que ganha por ter curvas e o que ganha por outras diferenças entre os dois modelos.

**Secção 6.4, o controlo de sobreajuste (nova).** Ver o ponto 5 abaixo.

**Conclusões.** Resumo em bullets.

---

## 4. O que mudei hoje e porquê

1. **Fui buscar as atualizações da main.** O André mudou a tarefa 5: o Lasso passou a escolher o valor de `C` por validação cruzada no treino, e não olhando para o teste. O `C` é o "quão severo é o Lasso": quanto mais pequeno, mais variáveis ficam a zero. Passou a ser 0,05, e o Lasso mantém 18 variáveis (antes o notebook usava `C = 0.1` e ficava com mais). A tarefa 6 deve usar as mesmas variáveis da tarefa 5, por isso atualizei-a.
2. **Corrigi um pormenor técnico no Lasso.** No scikit-learn 1.7 é preciso escrever `penalty="l1"`, senão o `l1_ratio` é ignorado e o modelo não é um Lasso. Já está explícito.
3. **Acrescentei gráficos**: curvas ROC e de precisão-recall, mapa de calor dos coeficientes do Lasso, barras com a importância de cada variável, gráfico de calibração, histogramas das probabilidades e matrizes de confusão no teste.
4. **Acrescentei a secção 6.4** sobre sobreajuste, a pedido do que o professor disse.
5. **Reescrevi o texto** que dependia dos números antigos, porque com outras variáveis os números mudaram (ver ponto 6).

---

## 5. A parte do sobreajuste e dos 10%

O professor falou (oralmente) de um limite de 10% e de o modelo não poder prever "tão bem" todos os pontos. Não encontrei esse número no enunciado nem nos slides, por isso **escrevi no notebook, de forma explícita, o critério que usei**: a perda relativa do treino para a validação (ou teste) tem de ser inferior a 10%. A conta é $(\text{treino} - \text{validação}) / \text{treino}$. Por exemplo, se o ROC-AUC é 0,99 no treino e 0,96 na validação, a perda é $0{,}03 / 0{,}99 \approx 3\%$. Se o professor se referia a outra coisa, a tabela permite recalcular.

Fiz quatro verificações:

1. **A diferença treino-validação-teste, nas três hipóteses.** O pior caso é 4,0% (o F1 de `HY vs resto`), no teste é no máximo 2,4%. Em pontos absolutos, no máximo 3,2. Está bem abaixo de 10%. Em vários casos o teste saiu até melhor do que o treino, o que é ruído normal com apenas 400 doentes.
2. **O efeito de λ.** Testei vários λ (mais rígidos e mais livres). A validação não piora quando o modelo fica mais livre; pelo contrário, melhora um pouco. Logo, o λ escolhido não é o que provoca sobreajuste.
3. **A curva de aprendizagem.** Treina-se com 128, 256, ... até 1280 doentes. Com poucos doentes o modelo generaliza pior (diferença de 3,2%); com todos, a diferença fecha (0,4%) e a validação ainda sobe. É o comportamento de um modelo que beneficia de mais dados, e não de um que decora.
4. **Os controlos.** Primeiro, baralhei as etiquetas dos doentes (agora não há nada a aprender). O modelo normal não aprende nada útil (ROC-AUC de 0,56 no treino e 0,52 na validação, perto do acaso). Se o modelo for obrigado a ser muito livre (λ fixo no valor mais baixo), já consegue decorar algum ruído (0,64 no treino), mas na validação fica no acaso (0,49). Ou seja, o modelo *tem* capacidade para sobreajustar, e a validação denuncia-o; com as etiquetas verdadeiras isso não acontece. Segundo, treinei um modelo só com a `HbA1c_level`, e deu ROC-AUC 0,982.

**Porque é que o desempenho em `DI` é tão alto (0,988)?** Porque a `HbA1c_level` quase decide sozinha: a mediana é 6,95 nos `DI` e 5,40 nos outros. Quando uma variável separa quase perfeitamente as classes, chama-se **quase separação**. As outras 15 variáveis só acrescentam 0,006 ao ROC-AUC. Não é sobreajuste, é um sinal muito forte nos dados. Nas outras duas hipóteses, onde não há uma variável assim, o ROC-AUC ronda 0,87, o que não é suspeito.

---

## 6. Os resultados principais

| Pergunta | Resposta |
|---|---|
| Que hipótese foi escolhida? | `DI vs resto`: ROC-AUC 0,988, contra 0,869 (`N`) e 0,868 (`HY`) |
| Que variável domina? | `HbA1c_level`, com um efeito de cerca de 39 unidades de logit, muito acima das outras (5,3 a 1,7) |
| A `HbA1c` tem efeito curvo? | Quase uma reta. A curva é estatisticamente necessária, mas só se afasta da reta entre 6 e 7,2 |
| Outras variáveis com efeito não linear? | Nenhuma individualmente. `alcohol_intake` tem forma de U (mesma média, mas maior dispersão nos `DI`) |
| GAM contra regressão logística, probabilidades | Log-loss 0,074 contra 0,103, melhor nas 15 divisões (significativo) |
| GAM contra regressão logística, discriminação | ROC-AUC 0,989 contra 0,982, melhor nas 15 divisões, mas sem significância (p = 0,053) |
| No teste (400 doentes, 66 `DI`) | ROC-AUC 0,984 (GAM) contra 0,978 (LR); os dois encontram 59 dos 66 `DI`; GAM com 2 falsos positivos, LR com 6 |
| Sobreajuste? | Perda máxima de 4,0% (limite 10%); controlos não mostram memorização |

Em resumo: para `DI`, uma regressão logística simples já discrimina muito bem. O GAM acrescenta um ganho moderado e consistente, sobretudo na qualidade das probabilidades. O custo é uma interpretação mais difícil e a necessidade de verificar o sobreajuste.

### Como ler "logit" e a amplitude

O logit é a escala em que os efeitos se somam. 0 significa "sem efeito"; +1 multiplica por cerca de 2,7 as hipóteses (odds) de ser `DI`; −1 divide por 2,7. A amplitude é a diferença entre o ponto mais alto e o mais baixo da curva. Um valor de 39 é enorme (uma variável que, só por si, muda a probabilidade de quase impossível a quase certa), e é o que caracteriza a quase separação.

---

## 7. Perguntas que o professor pode fazer (e respostas curtas)

**Porque escolheste `DI vs resto`?** Porque tem o maior ROC-AUC médio na validação cruzada (0,988), e esta métrica permite comparar as três hipóteses mesmo com classes de tamanhos diferentes. A diferença para as outras (0,869 e 0,868) é muito maior do que a variação entre divisões (cerca de 0,01).

**O que é um GAM e em que difere da regressão logística?** Ambos somam o efeito de cada variável. Na logística o efeito é uma reta; no GAM é uma curva estimada a partir dos dados. Se todas as curvas forem retas, é a mesma coisa.

**Como escolheste os preditores?** Usei o Lasso da tarefa 5, como pede o enunciado, com o `C = 0,05` escolhido por validação cruzada no treino. Ficaram 18 variáveis; retirei duas que são a idade escrita de outra forma, e ficaram 16.

**Para que serve o λ e como o escolheste?** Controla quão suave é a curva. Foi escolhido por `gridsearch` dentro de cada partição de treino, sem olhar à de validação. O valor escolhido (0,1) é o mais baixo da grelha, mas mostrei que valores ainda mais baixos não pioram a validação, por isso não há sobreajuste por causa disso.

**Que variáveis têm efeito não linear?** Só a `HbA1c_level`, de forma ligeira, é estatisticamente necessária. As outras cinco contínuas são compatíveis com retas. O `alcohol_intake` mostra um U, que se explica por os `DI` terem a mesma média mas maior dispersão.

**Isto é consistente com a EDA?** Sim. A `HbA1c` é a primeira variável nos dois rankings, e a correlação de Spearman entre as duas ordenações é 0,89. As diferenças são trocas de lugares vizinhos.

**Porque é que o GAM não é muito melhor do que a regressão logística?** Porque a `HbA1c` já separa quase tudo e é quase uma reta, e a regressão logística aproveita isso. O ganho do GAM está sobretudo na qualidade das probabilidades (log-loss), não na capacidade de ordenar doentes.

**Como sabes que não há sobreajuste?** Quatro razões: a perda treino-validação é no máximo 4%, e no teste 2,4%; a validação não piora com modelos mais livres; a curva de aprendizagem fecha com mais dados; e com etiquetas baralhadas o modelo fica ao nível do acaso na validação.

**Porque é que o desempenho em `DI` é tão alto?** Por causa da `HbA1c`, que sozinha dá ROC-AUC 0,982. É um sinal real e forte, não memorização.

**Porque não usaste os p-values do `pygam`?** A própria biblioteca avisa que são demasiado otimistas quando o λ é estimado. Em vez disso, comparei modelos na validação cruzada, com um teste t corrigido que tem em conta que as divisões partilham doentes, e com a correção de Bonferroni.

**O que é Bonferroni?** Quando se fazem muitos testes, alguns dão "significativo" por sorte. A correção torna o critério mais exigente na proporção do número de testes.

**O que são os 10%?** Foi uma indicação oral do professor. Interpretei-a como "a perda relativa treino-validação não pode passar 10%", defini-o no notebook, e o modelo cumpre com folga.

---

## 8. Limites honestos (podem ser perguntados)

- Os preditores foram escolhidos com o treino todo, e depois avaliados por validação cruzada dentro do treino. Isto torna os valores ligeiramente otimistas em termos absolutos (por igual para todos os modelos, por isso a comparação continua justa).
- O conjunto de teste tem só 66 `DI`: diferenças pequenas em contagens (um falso positivo a mais ou a menos) não dizem muito.
- Os intervalos de confiança e p-values do `pygam` são aproximados.
- Há uma repetição estranha de valores exatamente iguais em várias variáveis (por exemplo, 88 doentes com `blood_pressure` igual), que a EDA já assinalava. Pode ser imputação pela mediana e convém confirmar a origem.
- O valor de λ escolhido é o limite da grelha.

---

## 9. O que ainda falta

1. Confirmar com o professor o que significa exatamente o limite de 10%.
2. Instalar o `pygam` no teu ambiente (`pip install pygam`), porque o kernel que usas no VS Code não o tem.
3. Decidir se se faz commit e push do `task6_GAM.ipynb` para a main (ainda não fiz).
4. Decidir se se atualiza o notebook da `Parte1` com o Ridge/Lasso novo do André; esse notebook ainda tem a versão antiga e texto desatualizado.

---

## Glossário rápido

| Termo | Significado |
|---|---|
| One-vs-Rest | Um modelo por classe, "esta classe ou qualquer outra" |
| GAM | Modelo que soma efeitos de cada variável, cada efeito pode ser curva |
| Spline | A curva suave usada para estimar o efeito |
| λ | Travão da curva: maior é mais suave |
| EDoF | "Quanta curva" o modelo usa (1 é reta) |
| Lasso | Regressão que põe a zero os coeficientes das variáveis pouco úteis |
| Validação cruzada | Dividir o treino em partes e treinar/testar alternadamente |
| ROC-AUC | Probabilidade de ordenar bem um positivo e um negativo (0,5 acaso, 1 perfeito) |
| Log-loss | Qualidade das probabilidades (menor é melhor) |
| Quase separação | Uma variável separa quase perfeitamente as classes |
| Sobreajuste | O modelo decora o treino e generaliza mal |
| Bonferroni | Correção para quando se fazem muitos testes |
| Calibração | As probabilidades previstas correspondem às frequências reais |
