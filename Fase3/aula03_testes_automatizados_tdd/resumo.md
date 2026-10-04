# Aula 03 — Testes Automatizados

## Visão Geral
Terceira aula da disciplina "Engenharia de Software para Cientista de Dados" (Fase 3). O material trata de **testes automatizados** aplicados a projetos de ciência de dados: por que código exploratório em notebooks tende a quebrar sem manutenção, como testes unitários e de integração previnem regressões, e como a prática de **Test-Driven Development (TDD)** profissionaliza o desenvolvimento. A aula combina uma sessão prática com Pytest (hands-on) e uma seção teórica extensa ("Saiba Mais") cobrindo níveis de teste, boas práticas, considerações de recursos (CPU/GPU/memória) e peculiaridades de testar modelos de Machine Learning e pipelines de Big Data.

## Tópicos Abordados
- Introdução: por que testes automatizados importam em ciência de dados
- Hands On: pipeline simples (`filtra_negativos` + `calcula_media`) testado com Pytest, incluindo teste de integração
- Tipos de teste: unitário, integração, end-to-end (E2E) — e a Pirâmide de Testes
- Testes de regressão, de desempenho (performance) e de schema/qualidade de dados
- Cobertura de testes e onde focar esforço
- TDD (Test-Driven Development): ciclo Red/Green/Refactor e sua aplicação (e limitações) em ciência de dados
- 13 boas práticas de testes para projetos de ciência de dados (AAA, determinismo, fixtures, mocking, parametrização etc.)
- Considerações sobre recursos: memória, CPU/GPU, ambiente e testes distribuídos/concorrentes
- Casos especiais: testando modelos de Machine Learning e pipelines de Big Data/ETL
- Mercado, cases e tendências (Knight Capital, Great Expectations, Deepchecks etc.)

## Conceitos-Chave

### Tipos de Teste
- **Testes unitários**: testam unidades de código isoladamente (ex.: uma função de limpeza de dados). São *white-box*, cobrem caminho feliz e casos de borda, rápidos e baratos — podem rodar a cada commit.
- **Testes de integração**: avaliam a interação entre módulos combinados (ex.: carregar → transformar → rodar modelo → salvar). São mais *black-box*, mais lentos e custosos, e evitam a síndrome do "funciona na minha máquina".
- **Testes end-to-end (E2E)**: percorrem o sistema do início ao fim em ambiente próximo do real (ex.: reprocessar um lote real de dados em staging). São os mais lentos e caros, usados estrategicamente antes de releases importantes.
- **Pirâmide de Testes**: muitos testes unitários rápidos na base, alguns de integração no meio, poucos E2E no topo.
- Outros tipos citados: **testes de regressão** (garantir que mudanças não reintroduzam bugs antigos), **testes de desempenho** (tempo/memória, ex. `pytest-benchmark`) e **testes de schema/qualidade de dados** (ex. Great Expectations, Deequ — validam os dados, não o código).

### TDD (Test-Driven Development)
Técnica em que o teste é escrito **antes** da implementação. Ciclo **Red → Green → Refactor** (Nelson, 2024):
1. **Red**: escreva um teste que falha, pois a funcionalidade ainda não existe.
2. **Green**: implemente o código mínimo para o teste passar (sem se preocupar com elegância).
3. **Refactor**: com o teste verde como rede de segurança, melhore a estrutura do código.

Benefícios: clarifica requisitos ("dada entrada X, espero saída Y"), força interfaces mais simples e testáveis, e reduz acoplamento. **Limitações em ciência de dados** (segundo Catherine Nelson): na fase exploratória, o caminho até a solução é incerto e o código pode ser descartado — TDD estrito nem sempre compensa nessa fase; é mais útil quando o código começa a ser consolidado para produção. Em DS, muitas vezes não se sabe o resultado numérico exato, então o TDD se adapta para testar **propriedades e invariantes** (ex.: `norm(normaliza(v)) == 1`), usando inclusive frameworks de *property-based testing* como Hypothesis.

### Boas Práticas de Testes em Ciência de Dados
1. **Arrange-Act-Assert (AAA)**: estruturar cada teste em preparação, ação e verificação.
2. **Testes determinísticos**: fixar seeds (`random.seed(42)`) para evitar falsos negativos por aleatoriedade; para funções estocásticas, testar propriedades estatísticas (ex.: qui-quadrado) em vez de valores exatos.
3. **Fixtures e dados sintéticos pequenos**: montar dataframes/arrays pequenos cobrindo valores normais, de borda e formatos alternativos; limpar efeitos colaterais (arquivos temporários etc.).
4. **Evitar dependência entre testes**: cada teste deve rodar isoladamente, em qualquer ordem, permitindo paralelização (`pytest -n auto`).
5. **Nomear claramente os testes** (ex.: `test_filtra_negativos_remove_apenas_negativos`).
6. **Parametrização e markers do Pytest**: `@pytest.mark.parametrize`, `@pytest.mark.slow`, `@pytest.mark.skipif` para manter a suíte eficiente.
7. **Mocking criterioso**: simular I/O de rede/banco, tempo/aleatoriedade e funções pesadas — mas sem exagerar (não mockar modelos em testes de integração, que devem validar o comportamento real).
8. **Testar casos de erro**: verificar que inputs inválidos geram exceções claras (`pytest.raises(TypeError)`), não falhas silenciosas.
9. **Monitorar cobertura com sabedoria**: usar `coverage.py`, mas focar em lógica crítica em vez de perseguir 100% de forma cega.
10. **Equilibrar amostras pequenas vs. realistas**: usar datasets reduzidos para manter os testes rápidos, sem perder a representatividade da lógica.
11. **Validar contra implementações de referência**: comparar algoritmo próprio com `scikit-learn` ou verificar invariantes matemáticas/formais (ex.: lista ordenada e é permutação da original).
12. **Testar código de infraestrutura e utilitários**: validar arquivos de configuração (YAML/JSON) e scripts de ETL.
13. **Integrar testes à CI/CD**: rodar a suíte a cada push/PR (GitHub Actions, GitLab CI etc.), usando os testes como *gate* de deploy.

### Testando Modelos de Machine Learning
- **Sanidade no treinamento**: testar se o modelo consegue dar *overfit* em um conjunto minúsculo de dados (ex.: 10 amostras) — se não conseguir, há bug no treino/gradiente.
- **Comparação com implementação de referência**: validar algoritmo próprio contra bibliotecas confiáveis (ex.: `scikit-learn`).
- **Invariantes do modelo**: ex. saída de sigmoide deve estar em (0,1); ausência de `NaN`; todo ponto deve ser atribuído a algum cluster no K-means.
- **Persistência de modelo**: modelo salvo (`joblib.dump`) e recarregado deve produzir as mesmas previsões.
- **Vieses e equidade (fairness)**: testar se métricas (ex.: taxa de falsos positivos) não divergem excessivamente entre subgrupos demográficos.

### Testando Pipelines de Big Data/ETL
- Usar motores de processamento em **modo local** (ex.: Spark `local[*]`) com dados pequenos para testes de integração.
- Injetar dados problemáticos (CSV corrompido) para validar tratamento de erro; usar *monkeypatch* para simular falhas (ex.: banco indisponível) e checar retries/logs.
- **Smoke tests**: execução simplificada (`dry_run=True`) que verifica se dependências, caminhos e credenciais estão ok, sem validar resultados em detalhe.
- Cuidados com **concorrência e thread-safety**: GIL do Python, uso de `multiprocessing`, ordenação não determinística de saídas paralelas.

### Recursos: Memória, CPU/GPU e Ambiente
- Monitorar vazamentos de memória com `tracemalloc` ou `memory_profiler`.
- Verificar uso real de GPU (ex.: `tensor.device == "cuda:0"` em PyTorch).
- Rodar testes em múltiplos ambientes (versões de Python, SO) para pegar problemas de portabilidade (ex.: separadores de path).

## Exercício Hands-On (do material)
Pipeline simples de ciência de dados com duas funções e testes em Pytest, aplicando o espírito do TDD:

- `filtra_negativos(valores)`: remove valores negativos de uma lista.
- `calcula_media(valores)`: calcula a média aritmética, retornando `0` para lista vazia (evita divisão por zero).
- Testes unitários parametrizados (`@pytest.mark.parametrize`) para cada função, cobrindo casos normais, apenas negativos e lista vazia.
- Um teste de integração (`test_pipeline_completo`) que encadeia as duas funções e confere tanto o resultado intermediário quanto o final.

## Exemplos de Código

Implementação das funções do pipeline:

```python
# Implementação das funções alvo do pipeline
def filtra_negativos(valores):
    """Retorna uma lista contendo apenas os valores não-negativos de 'valores'."""
    return [x for x in valores if x >= 0]

def calcula_media(valores):
    """Retorna a média aritmética dos valores da lista, ou 0 se a lista for vazia."""
    if not valores:  # lista vazia
        return 0
    return sum(valores) / len(valores)
```

Testes unitários e de integração com Pytest:

```python
# test_pipeline.py – arquivo de testes
import pytest
from pipeline import filtra_negativos, calcula_media

# Teste unitário para filtra_negativos
@pytest.mark.parametrize(
    "entrada, esperado",
    [
        ([-1, 5, 0, -3, 2], [5, 0, 2]),   # mistura de negativos e não-negativos
        ([1, 2, 3], [1, 2, 3]),           # nenhum negativo a remover
        ([-5, -6], []),                   # só negativos, resultado vazio
        ([], []),                         # lista vazia permanece vazia
    ]
)
def test_filtra_negativos(entrada, esperado):
    assert filtra_negativos(entrada) == esperado

# Teste unitário para calcula_media
@pytest.mark.parametrize(
    "entrada, esperado",
    [
        ([5, 5, 5], 5),   # média de valores iguais
        ([2, 4], 3),      # média inteira exata
        ([2, 3, 4], 3),   # média não-inteira
        ([], 0),          # lista vazia -> 0 (caso especial)
    ]
)
def test_calcula_media(entrada, esperado):
    assert calcula_media(entrada) == esperado

# Teste de integração do pipeline completo (filtragem seguida de média)
def test_pipeline_completo():
    dados_brutos = [-1, 2, 4, -3]
    filtrado = filtra_negativos(dados_brutos)
    resultado_final = calcula_media(filtrado)
    assert filtrado == [2, 4]
    assert resultado_final == 3
```

Padrão Arrange-Act-Assert (AAA):

```python
# Arrange
texto = "hello"
esperado = 2
# Act
resultado = conta_vogais(texto)
# Assert
assert resultado == esperado
```

Teste de comportamento esperado em condições inválidas:

```python
import pytest

def test_calcula_media_input_invalido():
    with pytest.raises(TypeError):
        calcula_media("texto em vez de lista")  # deve disparar TypeError
```

## Cases e Tendências de Mercado

**Knight Capital (2012) — Bug de US$ 440 milhões**
Em agosto de 2012, um erro de software quase levou a Knight Capital à falência em 45 minutos, causando prejuízo de US$ 440 milhões após execuções erradas de código legado. A falta de testes adequados (regressão e end-to-end) permitiu o problema. O caso virou referência sobre a importância de testes abrangentes e protocolos seguros no setor financeiro, levando a práticas como deploy gradativo e simulações antes de mudanças.

O livro *Test-Driven Development with Python* (Harry Percival), embora focado em web, inspirou muitos a usarem TDD; sua 3ª edição (2023) discute TDD na era da IA e ressalta que a prática continua relevante para produzir "código limpo que funciona".

Blogs como o de **Eugenia Yan** (ex-engenheira da Amazon) trazem artigos como "Don't Mock ML Models in Unit Tests" e "Writing Robust Tests for Data & ML Pipelines", com dicas práticas e exemplos de código Pytest para pipelines de recomendação.

Ferramentas open source em ascensão: **Great Expectations** e **Deequ** (qualidade de dados), **Deepchecks** (testes de performance e fairness para modelos de ML) e **Checklist** (Microsoft Research, para testar comportamentos de modelos de NLP). Exemplo citado: a empresa **Calm** integrou verificações do Great Expectations em seu fluxo de ETL no Airflow, recebendo alertas no Slack quando uma tabela recebida tem menos linhas que o esperado.

## Checklist de Estudo
- [ ] Sei diferenciar testes unitários, de integração e end-to-end (E2E) e entendo a Pirâmide de Testes
- [ ] Sei explicar o ciclo Red/Green/Refactor do TDD e quando ele vale (ou não) a pena na fase exploratória de DS
- [ ] Sei escrever testes parametrizados com `@pytest.mark.parametrize` e testes de integração encadeando funções
- [ ] Sei estruturar um teste no padrão Arrange-Act-Assert (AAA)
- [ ] Sei fixar seeds para testes determinísticos e testar propriedades estatísticas quando a aleatoriedade não pode ser removida
- [ ] Sei quando e como usar mocking (I/O, tempo, funções pesadas) sem perder o sentido do teste
- [ ] Sei testar casos de erro/inputs inválidos com `pytest.raises`
- [ ] Sei o que são testes de schema/qualidade de dados (Great Expectations, Deequ) e por que complementam testes de código
- [ ] Sei as estratégias específicas para testar modelos de ML (overfit em dados minúsculos, invariantes, persistência, fairness)
- [ ] Sei o que são smoke tests e como validam ambiente/configuração em pipelines de ETL
- [ ] Conheço o caso Knight Capital e por que ele reforça a importância de testes de regressão/E2E

## Palavras-chave
Testes Automatizados. Engenharia de Software para Ciência de Dados. Test-Driven Development (TDD).

## Referências
- CHOREV, S. **Top 10 ML Model Failures You Should Know About**. 2022. Disponível em: https://www.deepchecks.com/top-10-ml-model-failures-you-should-know-about/. Acesso em: 12 dez. 2025.
- GREAT EXPECTATIONS. **How Calm uses GX to create data quality alerts and avert critical data issues in Airflow DAGs**. 2024. Disponível em: https://greatexpectations.io/case-studies/how-calm-uses-gx-to-create-data-quality-alerts-and-avert-critical-data/. Acesso em: 12 dez. 2025.
- NELSON, C. **Software Engineering for Data Scientists**. Sebastopool: O'Reilly Media, 2024. Disponível em: https://unidel.edu.ng/focelibrary/books/Software%20Engineering%20for%20Data%20Scientists%20%28Catherine%20Nelson%29%20%28Z-Library%29.pdf/. Acesso em: 12 dez. 2025.
- PERCIVAL, H. **Test-Driven Development with Python**. 3. ed. Sebastopool: O'Reilly Media, 2025.
- YAN, E. **Writing Robust Tests for Data & Machine Learning Pipelines**. 2023. Disponível em: https://eugeneyan.com/writing/testing-pipelines/. Acesso em: 12 dez. 2025.
