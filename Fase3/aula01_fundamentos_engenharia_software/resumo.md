# Aula 01 — Fundamentos de Engenharia de Software

## Visão Geral
Aula introdutória da disciplina de Engenharia de Software para Cientistas de Dados (Fase 3). Parte da constatação de que mais de 80% dos projetos de ciência de dados falham ou nunca geram valor de negócio, muitas vezes por problemas na transição do "código de laboratório" (notebooks experimentais) para sistemas robustos em produção. Defende que bons modelos precisam vir acompanhados de bom código — organizado, reprodutível e sustentável — e que a Engenharia de Software fornece a disciplina necessária para isso, convergindo com a Ciência de Dados na prática de **MLOps**.

## Tópicos Abordados
- Por que ciência de dados precisa de engenharia de software (dívida técnica, reprodutibilidade)
- Hands On: refatoração de um código de análise de dados desorganizado em uma função modular e testável
- Ciclo de vida de desenvolvimento: Waterfall vs. Agile e o modelo CRISP-DM / TDSP
- Princípios de design de código: modularidade, coesão e acoplamento
- Princípio da Inversão de Dependência e Injeção de Dependência (o "D" de SOLID)
- Boas práticas: controle de versão (Git), documentação, testes automatizados e code review
- Manutenibilidade e dívida técnica em sistemas de Machine Learning
- Paradigmas de programação: procedural, orientado a objetos e funcional
- Estruturas de dados, complexidade computacional (Big O) e escalabilidade
- Mercado, cases e tendências em MLOps

## Conceitos-Chave

### Dívida Técnica em ML
Analogia de Ward Cunningham: atalhos no código são como dívidas que cobram juros no futuro. O artigo do Google *"Machine Learning: The High-Interest Credit Card of Technical Debt"* (Sculley et al., 2015) alerta que sistemas de ML acumulam dívida técnica de forma ainda mais insidiosa que softwares tradicionais, pois incluem componentes extras (dados, modelos, pipelines). Um estudo do Google mostrou que o código do modelo em si costuma representar **menos de 5%** do total de um sistema de ML maduro — os outros 95% são infraestrutura e código auxiliar (ETL, validação, monitoramento).

### Ciclo de Vida: Waterfall, Agile e CRISP-DM
O ciclo clássico de desenvolvimento (concepção → desenvolvimento → testes → implantação → manutenção) pode seguir o modelo **Waterfall** (sequencial, rígido, pouco compatível com a natureza exploratória da ciência de dados) ou o modelo **Agile** (iterativo, em sprints curtos, com feedback contínuo — mais adequado quando não se sabe de antemão qual modelo funcionará). O **CRISP-DM** (Cross-Industry Standard Process for Data Mining) formaliza um ciclo cíclico específico para dados: Entendimento do Negócio → Entendimento dos Dados → Preparação dos Dados → Modelagem → Avaliação → Deploy, com retorno a etapas anteriores quando necessário. A Microsoft propôs o **TDSP** (Team Data Science Process), que adiciona práticas de DevOps ao CRISP-DM, convergindo para a noção de **MLOps**.

### Modularidade, Coesão e Acoplamento
- **Modularidade**: dividir o software em componentes menores com responsabilidades claras ("dividir para conquistar"), facilitando teste isolado, reuso e trabalho em paralelo entre times.
- **Coesão**: o quanto as partes internas de um módulo convergem para um único propósito. Alta coesão = módulo "especialista" em uma coisa só (princípio da responsabilidade única); baixa coesão = módulo que acumula responsabilidades não relacionadas (cálculo + visualização + I/O + notificação, por exemplo).
- **Acoplamento**: grau de interdependência entre módulos. Alto acoplamento = módulos fortemente amarrados (difícil de mudar e de testar isoladamente); baixo acoplamento = módulos que se comunicam só pelo necessário, via interfaces estáveis.

### Princípio da Inversão de Dependência e Injeção de Dependência
Quando um módulo de alto nível depende diretamente de uma implementação concreta (ex.: `ProcessadorDePedidos` instanciando `ServicoDeEmail` internamente), o código fica difícil de mudar (viola o princípio Aberto/Fechado) e difícil de testar (não dá para isolar a lógica de negócio do envio real de e-mail). A solução é o **"D" de SOLID**: ambos os módulos devem depender de uma abstração (uma interface/classe abstrata). Na prática, isso se resolve com **Injeção de Dependência** — a dependência concreta é passada no construtor (`__init__`) em vez de ser criada internamente, permitindo trocar a implementação (e-mail, SMS, Slack) ou substituí-la por um **mock** em testes unitários, sem alterar a classe principal.

### Boas Práticas: Versionamento, Documentação, Testes e Revisão
- **Controle de versão (Git)**: substitui o caos de arquivos tipo `analise_final_v2_bkp.py`; permite branches, histórico, reversão e integração via pull requests. Também se estende a dados e modelos (ex.: DVC — Data Version Control), essencial para reprodutibilidade.
- **Documentação**: código autoexplicativo (nomes claros, PEP8) + docstrings + README + registro de decisões e hipóteses (por que um outlier foi removido, por que um modelo foi escolhido).
- **Testes automatizados**: testes unitários (ex.: `calcular_media([1,2,3,None]) == 2`), testes de integração (rodar o pipeline inteiro com um subconjunto de dados) e **testes de regressão** (alertar se um novo modelo piorou uma métrica de referência). Ferramentas: `unittest`, `pytest`.
- **Code review**: "quatro olhos veem mais que dois" — compartilha conhecimento, padroniza código e detecta bugs/casos não tratados.

### Manutenibilidade
Resulta diretamente da aplicação de modularidade, testes, documentação e versionamento. Um código mal estruturado frequentemente leva equipes a descartar a "versão 1" de um sistema e recomeçar do zero, por causa de dívidas técnicas não pagas (dependências de dados instáveis, falta de monitoramento, efeitos colaterais entre componentes — Sculley et al., 2015).

### Paradigmas de Programação
| Paradigma | Quando usar | Vantagens | Desvantagens |
|---|---|---|---|
| Procedural | Scripts simples, protótipos rápidos | Fácil de entender e implementar | Difícil de escalar e manter |
| Orientado a Objetos (POO) | Projetos com múltiplas entidades e estados complexos | Modularidade, reutilização | Pode ser verboso e complexo |
| Funcional | Processamento de dados, pipelines, código declarativo | Concisão, menos efeitos colaterais | Pode ser menos intuitivo |

### Estruturas de Dados, Complexidade e Escalabilidade
Conhecer a complexidade computacional (Big O) do código é essencial: um algoritmo O(n²) pode ser inviável para 1 milhão de registros (1e12 operações), enquanto O(n log n) ou O(n) são tratáveis. Exemplo prático: buscar um item em uma **lista** é O(n); em um **set**/**dict** é O(1) amortizado. Bibliotecas vetorizadas (NumPy, Pandas) usam código C otimizado internamente e são ordens de magnitude mais rápidas que loops Python puros. Para big data, ferramentas como Apache Spark, Dask e Ray exigem código sem estado global compartilhado e alinhado a paradigmas funcionais.

## Exercício Hands-On (do material)
Refatoração de um pequeno código de análise de dados, originalmente desorganizado, em uma função modular, documentada e com tratamento de exceções — exemplo de cálculo de média de uma lista ignorando valores `None`. O material evolui o exemplo em camadas progressivas: (1) função simples com docstring e tratamento de caso excepcional (lista vazia); (2) separação de um script monolítico (carregar → limpar → treinar → avaliar) em módulos (`utils_dados.py`, `utils_modelo.py`, `main.py`); (3) refatoração de uma classe `ProcessadorDePedidos` fortemente acoplada a `ServicoDeEmail` para uma versão com Injeção de Dependência, testável com um `Mock`.

## Exemplos de Código

Função com modularidade, docstring e tratamento de caso excepcional:

```python
def calcular_media(valores):
    """Calcula a média de uma lista de números, ignorando valores None."""
    filtrados = [x for x in valores if x is not None]
    if len(filtrados) == 0:
        return None  # Evita divisão por zero se lista vazia após filtrar
    return sum(filtrados) / len(filtrados)

dados = [10, None, 25, 40]
media = calcular_media(dados)
print(f"Média calculada: {media:.2f}" if media is not None else "Lista vazia após filtragem.")
```

Pipeline modularizado (divisão em arquivos por responsabilidade):

```python
# utils_dados.py — Módulo focado em dados
import pandas as pd

def carregar_dados(caminho):
    try:
        return pd.read_csv(caminho)
    except FileNotFoundError:
        print(f"Erro: Arquivo {caminho} não encontrado.")
        return None

def limpar_dados(df):
    df_limpo = df.dropna()
    df_limpo['nova_coluna'] = df_limpo['coluna_existente'] * 2
    return df_limpo
```

```python
# utils_modelo.py — Módulo focado em ML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def treinar_modelo(X, y):
    modelo = LogisticRegression()
    modelo.fit(X, y)
    return modelo

def avaliar_modelo(modelo, X_test, y_test):
    preds = modelo.predict(X_test)
    return accuracy_score(y_test, preds)
```

Baixo acoplamento com Injeção de Dependência (Princípio da Inversão de Dependência):

```python
from abc import ABC, abstractmethod

class IServicoNotificacao(ABC):
    """Abstração: define o contrato que qualquer notificador deve seguir."""
    @abstractmethod
    def enviar_notificacao(self, destinatario, assunto, corpo):
        pass

class ServicoDeEmail(IServicoNotificacao):
    def enviar_notificacao(self, destinatario, assunto, corpo):
        print(f"[E-MAIL] Enviando para {destinatario}: {assunto}")
        return True

class ProcessadorDePedidos:
    """Fracamente acoplado: depende só da abstração, não de uma implementação concreta."""
    def __init__(self, servico_notificacao: IServicoNotificacao):
        self._notificador = servico_notificacao

    def processar_pedido(self, pedido):
        print(f"Processando pedido {pedido['id']}...")
        self._notificador.enviar_notificacao(
            pedido['cliente_contato'], "Seu pedido foi confirmado!", f"Detalhes: {pedido['id']}"
        )

# Trocar de e-mail para SMS não exige mudar ProcessadorDePedidos, só injetar outro serviço.
```

Exemplo de alta coesão (responsabilidade única por função):

```python
def calcular_estatisticas(lista_numeros):
    """Função focada 100% em cálculo."""
    media = sum(lista_numeros) / len(lista_numeros)
    mediana = sorted(lista_numeros)[len(lista_numeros) // 2]
    return {"media": media, "mediana": mediana}
```

## Cases e Tendências de Mercado
- **Case de fracasso**: em 2020, o governo do Reino Unido perdeu quase 16 mil registros de COVID-19 por usar Excel além do seu limite técnico, sem planejamento de engenharia — exemplo real do custo de improvisos em sistemas de dados.
- **Tendência de sucesso**: empresas como Netflix e Uber investem em plataformas de MLOps (ex.: Metaflow) para reprodutibilidade, versionamento e automação de pipelines, tornando projetos escaláveis e confiáveis.
- **Educação**: universidades e publicações (Harvard Data Science Review, Udacity, Coursera) vêm enfatizando boas práticas de codificação para cientistas de dados.
- **Futuro**: crescimento de frameworks e ferramentas de DataOps/observabilidade (ex.: Cookiecutter Data Science) e especialização de papéis (engenheiro de dados vs. cientista de dados).

## Checklist de Estudo
- [ ] Sei explicar por que dívida técnica em ML é mais insidiosa do que em software tradicional
- [ ] Sei diferenciar Waterfall de Agile e explicar por que CRISP-DM/TDSP se encaixam melhor em ciência de dados
- [ ] Sei definir modularidade, coesão e acoplamento e identificar exemplos de baixa/alta coesão no código
- [ ] Sei aplicar Injeção de Dependência para reduzir acoplamento e viabilizar testes com mocks
- [ ] Sei listar as boas práticas essenciais (Git, documentação, testes, code review) e seu papel na manutenibilidade
- [ ] Sei diferenciar os paradigmas procedural, orientado a objetos e funcional e quando usar cada um
- [ ] Sei estimar complexidade computacional (Big O) e escolher estruturas de dados adequadas (lista vs. set/dict)
- [ ] Conheço cases de mercado que ilustram o custo de más práticas e o valor de MLOps

## Palavras-chave
Reprodutibilidade · Escalabilidade · Manutenibilidade

## Referências
- KELION, L. *Excel: Why using Microsoft's tool caused Covid-19 results to be lost*. 2020. https://www.bbc.co.uk/news/technology-54423988
- LIFE CYCLE. *What is TDSP?* 2025. https://www.datascience-pm.com/tdsp/
- MASOLO, C. *Netflix enhances Metaflow with new configuration capabilities*. 2025. https://www.infoq.com/news/2025/01/netflix-metaflow-configuration/
- MUNSHI, A. *Why 80% of Data Science Projects Fail — And How to Fix It*. 2025.
- NELSON, C. *Software Engineering for Data Scientists: From Notebooks to Scalable Systems*. Sebastopol: O'Reilly Media, 2024.
- PRUIM, R.; GÎRJĂU, M.; HORTON, N. J. *Fostering Better Coding Practices for Data Scientists*. Harvard Data Science Review, v. 5, n. 3, 2023. https://hdsr.mitpress.mit.edu/pub/8wsiqh1c
- SCULLEY, D. et al. *Hidden Technical Debt in Machine Learning Systems*. 2015. https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf
- WIKIPEDIA COMMONS. *File:CRISP-DM Process Diagram.png*. 2012.
- WILSON, B. *Machine Learning Engineering in Action*. Shelter Island: Manning Publications, 2022.
