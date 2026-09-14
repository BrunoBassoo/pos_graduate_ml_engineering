# Aula 07 — Governança e Ciclo de Feedbacks

## Visão Geral
Aula da Fase 1 (ML Engineering) sobre como garantir que modelos de ML sejam usados com segurança, ética e em conformidade legal ao longo de todo o ciclo de vida. Aborda a estruturação da governança de modelos (versionamento, segurança, auditabilidade), as regulações e princípios éticos que a moldam (LGPD, AI Act, fairness, explicabilidade, accountability) e a necessidade de um ciclo de feedback contínuo (coleta de ground truth, detecção de drift, retreinamento) para manter a eficácia dos modelos em produção. Apresenta ferramentas práticas como MLflow e Fairlearn, além de boas práticas de documentação (Model Cards) e papéis organizacionais envolvidos.

## Tópicos Abordados
- O que vem por aí: por que governar modelos de IA é necessário mesmo após um deploy bem-sucedido
- Hands On: triagem de propostas de crédito com atenção a atributos sensíveis (região) usando Fairlearn
- Saiba Mais:
  - O que define um atributo como sensível (proteção legal/regulatória e risco ético/social)
  - Fairlearn: avaliação (MetricFrame) e mitigação de viés
  - Model Cards: transparência, gestão de risco, accountability e monitoramento contínuo
  - Model Card aplicado ao exemplo de aprovação de crédito (uso pretendido, limitações, avaliação de fairness)
  - Ciclo de Feedback Contínuo no ciclo de vida do modelo
  - Governança de modelos (versionamento, rastreabilidade/linhagem, segurança, responsabilidade, reprodutibilidade)
  - Regulações e Ética em IA (LGPD, AI Act, fairness, explicabilidade, accountability)
  - Ciclo de Feedbacks no Ciclo de Vida (coleta de ground truth, reavaliação periódica, aprendizado contínuo, detecção de drift, retreinamento, retroalimentação de stakeholders)
  - Ferramentas Práticas para Governança e Auditoria (MLflow, Fairlearn, Model Cards, AI FactSheets, plataformas de MLOps)
  - Boas Práticas de Auditoria, Validação e Documentação
  - Papéis e Responsabilidades no Ciclo de Governança
- Mercado, Cases e Tendências: mercado de governança de IA, caso Fairlearn (Microsoft + EY), Responsible AI nas Big Techs, tendências organizacionais (governance as code, CDAO, profissionalização da função de AI Model Governance)

## Conceitos-Chave

### Governança de Modelos
Conjunto de processos, políticas e controles que regulam o ciclo de vida de modelos de ML para assegurar qualidade, transparência e compliance. Envolve quatro pilares principais: 1) versionamento e rastreamento de modelos e dados; 2) segurança e integridade do sistema; 3) responsabilidades claras em cada etapa; 4) reprodutibilidade e auditabilidade dos experimentos.

### Atributos Sensíveis
Características ou variáveis presentes nos dados de treinamento que representam grupos protegidos legal ou eticamente (ex.: gênero, raça, etnia, orientação sexual, religião, renda, CEP/localidade, condição de saúde). Sua utilização ou omissão inadequada pode levar a modelos que exibem viés ou discriminação injusta. São definidos por dois critérios: **proteção legal/regulatória** (uso que pode violar leis antidiscriminação) e **risco ético/social** (mesmo sem proibição legal explícita, geram preocupações de fairness e equidade).

### Viés de Disparidade
Ocorre quando um modelo apresenta métricas de desempenho (acurácia, taxa de falsos positivos, taxa de falsos negativos) significativamente diferentes entre subgrupos definidos por atributos sensíveis. A mitigação envolve identificar essas disparidades e aplicar técnicas de correção antes do deploy.

### Fairlearn
Biblioteca Python (Microsoft) para avaliar e mitigar viés em modelos de ML. O uso se dá em duas fases:
- **Avaliação de Viés (MetricFrame)**: define-se o atributo sensível e um conjunto de métricas de desempenho/fairness (acurácia, taxa de seleção, taxa de falso positivo, taxa de falso negativo). O objeto `MetricFrame` calcula automaticamente as **métricas por grupo** (`by_group`) e a **disparidade** (`difference()`) — a diferença absoluta máxima entre o subgrupo mais e o menos favorecido.
- **Mitigação de Viés**: se a disparidade for inaceitável, o Fairlearn oferece algoritmos de Pré-processamento, In-processing (ex.: Exponentiated Gradient) e Pós-processamento (ex.: Threshold Optimizer).

### Model Cards
Documentação estruturada ("ficha técnica" ou "rótulo nutricional") de um modelo, originalmente proposta pelo Google, com quatro funções principais:
- **Transparência e Auditoria**: padroniza a documentação para que auditores/reguladores verifiquem se as decisões foram tomadas de forma justa.
- **Gestão de Risco**: detalha Uso Pretendido e Limitações, prevenindo uso indevido (misuse).
- **Responsabilidade (Accountability)**: registra desenvolvedor, data de treinamento e dados usados, permitindo rastrear e corrigir falhas.
- **Monitoramento Contínuo (MLOps)**: funciona como um "contrato de desempenho", definindo limites de disparidade aceitáveis que, se violados em produção, disparam retreinamento, mitigação ou retirada do modelo.

Em um pipeline de MLOps, o Model Card se integra como **geração automática de compliance** (o pipeline de Continuous Training o preenche automaticamente) e como **gate de governança** no Continuous Deployment (se a disparidade for "Não Aceitável", o deploy é barrado, exigindo aprovação manual ou retreinamento — governança human-in-the-loop).

### Ciclo de Feedback Contínuo
Mecanismo que garante a sustentabilidade, relevância e conformidade dos sistemas de ML ao longo do tempo. Tem três funções essenciais:
1. **Prevenção/correção de Model Drift e Data Drift**: o ground truth coletado permite reavaliar o desempenho e acionar o Continuous Training (CT).
2. **Manutenção de Fairness e Conformidade Ética**: o Continuous Monitoring (CM) audita continuamente as métricas de fairness usando o novo ground truth.
3. **Melhoria Contínua e Accountability**: o ground truth é matéria-prima para o retreinamento, e a documentação rigorosa do ciclo estabelece linhagem (lineage) rastreável.

### Componentes do Ciclo de Feedback no Ciclo de Vida
- **Coleta de Ground Truth**: comparar cada predição com o resultado real observado posteriormente (ex.: clique do usuário, diagnóstico clínico final).
- **Reavaliação Periódica**: revisões agendadas (mensal, trimestral) recalculando métricas com dados atuais para detectar degradação antes de falhas críticas.
- **Mecanismos de Aprendizado Contínuo**: online learning (atualização a cada novo dado) ou batch retraining (job periódico que re-treina o modelo inteiro), exigindo validação automática a cada ciclo.
- **Detecção de Model e Data Drift**: monitoramento estatístico da distribuição dos dados de entrada para sinalizar mudanças abruptas que exigem reavaliação/retreinamento.
- **Re-treinamento e Versionamento**: quando o modelo não atende mais aos padrões mínimos, ele é retreinado e passa pelas mesmas etapas de teste, validação e implantação.
- **Retroalimentação dos Stakeholders**: feedback humano (usuários, especialistas) combinado a métricas numéricas e KPIs de negócio (conversão, ROI, satisfação).

### LGPD (Lei 13.709/2018)
Regula o tratamento de dados pessoais no Brasil. O Art. 20 garante ao titular dos dados o direito de receber informações claras sobre decisões automatizadas e de solicitar revisão humana. A ANPD recomenda documentar todo o ciclo de IA (fontes de dados, lógica do modelo, salvaguardas). Exige também bases legais (consentimento, legítimo interesse) e cuidados com dados sensíveis.

### AI Act (União Europeia)
Aprovado em 2024, classifica sistemas de IA por nível de risco. Sistemas de "alto risco" (ex.: diagnóstico médico, scoring de crédito) exigem avaliações de conformidade, gerenciamento de risco, documentação técnica completa e transparência. Tarefas de "risco inaceitável" (ex.: classificação de pessoas por características biométricas) são proibidas. Exige transparência, segurança, robustez, supervisão humana e mitigação de vieses; mesmo modelos de baixo risco devem informar que o usuário está interagindo com uma IA.

### Fairness, Explicabilidade e Accountability
- **Fairness**: evitar discriminação indevida; auditar continuamente métricas como false positive rate entre grupos e aplicar mitigação quando houver desvios.
- **Explicabilidade e Transparência**: segundo o NIST, IA responsável requer sistemas "accountable e transparentes"; práticas como feature importance, LIME e SHAP ajudam a interpretar o modelo.
- **Accountability**: mecanismos internos de governança (papéis claros, revisões formais, junta de revisão para mudanças importantes, plano de contingência) para responsabilizar decisões de IA.

### Governança de Modelos — Pilares Detalhados
- **Versionamento de Modelos**: controle de versão de modelos e dados (ex.: MLflow Model Registry), associando cada versão a data, autor e branch do pipeline.
- **Rastreabilidade e Linhagem (data lineage)**: metadados que documentam origem dos dados, transformações aplicadas, data de treinamento e artefato final.
- **Segurança e Integridade**: controles de acesso, criptografia em repouso, logs de auditoria, TLS em APIs de inferência, testes de robustez a ataques adversariais.
- **Responsabilidade e Accountability**: papéis definidos (Cientista de Dados, MLOps, Compliance) e processos formais antes da liberação em produção.
- **Reprodutibilidade e Auditabilidade**: ambiente computacional congelado (containers), versionamento de bibliotecas, sementes aleatórias fixas, relatórios automatizados de métricas.

### Ferramentas Práticas
- **MLflow**: plataforma open-source de ciclo de vida de ML. O Tracking registra parâmetros, métricas e artefatos de cada execução; o Model Registry nomeia versões e define fases (Staging, Produção). O Dataset Tracking rastreia a origem dos dados usados.
- **Fairlearn**: avaliação e mitigação de viés (assessment + post-processing).
- **Outras**: Model Cards (Google), AI FactSheets (IBM), plataformas de MLOps com componentes de governança (Kubeflow, TFX, Metaflow, AWS SageMaker).

### Boas Práticas de Auditoria, Validação e Documentação
- **Auditorias Regulares**: revisão periódica de código, logs, configurações de acesso e políticas de dados, com checklists (ex.: "foram testados casos de viés?").
- **Validação Cruzada e Testes de Stress**: simulação de cenários extremos (outliers, dados corrompidos) integrada a pipelines CI/CD.
- **Documentação Estruturada (Model Cards)**: propósito do modelo, datasets, arquitetura, métricas globais e segmentadas por grupo sensível, usos pretendidos/proibidos, riscos conhecidos.
- **Políticas Internas e Datasheets**: classificação de modelos (público/interno/confidencial) e documentação de origem/qualidade/limitações dos datasets.
- **Treinamento e Cultura**: workshops sobre ética, fairness e segurança para aumentar a adesão aos processos de governança.

### Papéis e Responsabilidades no Ciclo de Governança
- **Cientistas de Dados**: treinam e documentam o modelo, atuam em fairness, entregam model cards e relatórios no MLflow.
- **Engenheiros(as) de MLOps/DevOps**: constroem pipelines reprodutíveis, configuram CI/CD, monitoram logs de produção, garantem segurança de infraestrutura.
- **Equipe de Segurança da Informação/Compliance**: avaliam riscos regulatórios, exigem avaliações de impacto de privacidade (DPIA), auditam acessos.
- **Gestores(as) de Produto/Negócio e Stakeholders**: definem problema de negócio, priorizam features, avaliam tolerância a risco.
- **Comitê de Ética de IA**: grupo interdisciplinar que supervisiona projetos de alto impacto e delibera sobre casos controversos.
- A adoção de uma matriz **RACI** (Responsável, Aprovador, Consultado, Informado) ajuda a evitar "passar a bola adiante" entre esses papéis.

## Exercício Hands-On (do material)
Cenário proposto: desenvolvimento de um modelo para uma instituição financeira, com o objetivo de automatizar o processo de triagem de propostas de crédito de seus clientes, utilizando atributos como renda, idade e região. A instituição tem a preocupação específica de garantir que não haja tratamento injusto em função da localidade (região) dos clientes, para que todos tenham acesso igualitário a oportunidades e recursos.

Para exercitar os conceitos de Governança de Modelos de Machine Learning, o exercício propõe o uso do framework **Fairlearn** para garantir que possíveis vieses contidos nos dados de treinamento sejam devidamente tratados. Os passos indicados são:
1. Criar um dataset sintético.
2. Treinar um modelo **Random Forest** para previsão da aprovação de crédito.
3. Com o modelo treinado, analisar se existem diferenças significativas entre as predições do modelo considerando o atributo sensível "região".

O material reforça que atributos sensíveis são características que representam grupos protegidos legal ou eticamente, e que sua utilização (ou omissão inadequada) pode levar a modelos com viés ou discriminação injusta contra certos grupos — daí a necessidade de avaliar o modelo sob a lente desses atributos antes de colocá-lo em produção.

## Exemplos de Código

O código a seguir reproduz o trecho apresentado no PDF para avaliar fairness com `MetricFrame`, aplicado ao atributo sensível "região" do exercício de aprovação de crédito.

```python
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)
from sklearn.metrics import accuracy_score

# Extrair atributo sensível (região)
sensitive_feature = X_test['region']

# Criar MetricFrame para avaliar fairness
metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'selection_rate': selection_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_feature,
)

# Visualizar métricas por grupo
print("Métricas por grupo (região):")
print(metric_frame.by_group)

print("\nDiferença máxima entre grupos:")
print(metric_frame.difference())
```

O trecho abaixo ilustra a etapa de **mitigação de viés** citada na aula (Pós-processamento via Threshold Optimizer), usada quando a disparidade calculada acima é considerada "Não Aceitável", como no exemplo da Tabela 4 do material (disparidade de 12 pontos percentuais na taxa de falso negativo, acima da margem máxima de 10%):

```python
from fairlearn.postprocessing import ThresholdOptimizer

# Mitigador de pós-processamento: ajusta o limiar de decisão por subgrupo
mitigator = ThresholdOptimizer(
    estimator=modelo_random_forest,
    constraints="false_negative_rate_parity",  # alinhado ao exemplo da aula (FN)
    objective="accuracy_score",
    prefit=True,
)

mitigator.fit(X_train, y_train, sensitive_features=X_train['region'])

y_pred_mitigado = mitigator.predict(X_test, sensitive_features=X_test['region'])

# Recalcular o MetricFrame com as predições mitigadas para comparar a nova disparidade
metric_frame_mitigado = MetricFrame(
    metrics={'false_negative_rate': false_negative_rate},
    y_true=y_test,
    y_pred=y_pred_mitigado,
    sensitive_features=X_test['region'],
)
print("Disparidade de FN após mitigação:", metric_frame_mitigado.difference())
```

Operacionalizando o conceito de **Model Card** apresentado na aula (Tabelas 2, 3 e 4 do PDF) como um artefato estruturado e versionável, útil como gate de governança em pipelines de CI/CD:

```yaml
# model_card_credit_approval.yaml
nome_modelo: "Credit_Approval_v2.1"
desenvolvedor: "Time de Data Science, Banco FIAPPOSTECH"
data_treinamento: "2025-11-29"
algoritmo: "Random Forest Classifier"
versionamento: "MLflow Model Registry - Versao 5"

uso_pretendido: "Triagem inicial automatizada de propostas de crédito pessoal para clientes pessoa física."
limitacoes:
  - "Nao deve ser usado para propostas de credito PJ."
  - "Nao deve ser usado em regioes geograficas com menos de 500 propostas historicas no dataset de treino, devido a sub-representacao."

avaliacao_fairness:
  atributo_sensivel: "regiao"
  grupos: ["Norte", "Sul", "Leste", "Oeste"]
  metricas:
    acuracia:
      geral: 0.880
      grupo_favorecido: 0.895   # Sul
      grupo_desfavorecido: 0.850  # Norte
      disparidade_maxima: 0.045
      margem_aceitavel: 0.05
      status: "aceitavel"
    taxa_falso_negativo:
      geral: 0.150
      grupo_favorecido: 0.100   # Sul
      grupo_desfavorecido: 0.220  # Norte
      disparidade_maxima: 0.120
      margem_aceitavel: 0.10
      status: "nao_aceitavel"   # aciona gate de governanca

gate_governanca:
  bloqueia_deploy_se: "status == 'nao_aceitavel' em qualquer metrica de fairness"
  acao_requerida: "aprovacao manual ou retreinamento (human-in-the-loop)"
```

Exemplificando a **coleta de ground truth e detecção de drift** dentro do ciclo de feedback contínuo, conceito central da seção "Ciclo de Feedbacks no Ciclo de Vida":

```python
import pandas as pd
from scipy.stats import ks_2samp

def registrar_predicao(id_cliente, features, predicao, modelo_versao):
    """Instrumenta a API de inferência para armazenar entrada e saída,
    permitindo vincular a verdade futura (ground truth) mais tarde."""
    log_predicao = {
        "id_cliente": id_cliente,
        "features": features,
        "predicao": predicao,
        "modelo_versao": modelo_versao,
        "timestamp": pd.Timestamp.now(),
        "resultado_real": None,  # preenchido quando o ground truth chegar
    }
    salvar_em_tabela_producao(log_predicao)


def atualizar_ground_truth(id_cliente, resultado_real):
    """Chamado quando o desfecho real é conhecido (ex.: inadimplência
    observada 90 dias depois), fechando o ciclo de feedback."""
    atualizar_tabela_producao(id_cliente, resultado_real=resultado_real)


def detectar_data_drift(dados_treino: pd.Series, dados_producao: pd.Series, alpha=0.05):
    """Teste de Kolmogorov-Smirnov para sinalizar mudanca na distribuicao
    de uma feature de entrada entre treino e producao (data drift)."""
    estatistica, p_valor = ks_2samp(dados_treino, dados_producao)
    drift_detectado = p_valor < alpha
    if drift_detectado:
        print(f"Data drift detectado (p={p_valor:.4f}) - acionar reavaliacao/retreinamento (CT)")
    return drift_detectado
```

Por fim, um exemplo de **log de decisão para auditoria**, operacionalizando os conceitos de accountability, rastreabilidade e linhagem descritos na aula:

```python
import json
from datetime import datetime, timezone

def registrar_decisao_auditavel(
    id_cliente,
    modelo_nome,
    modelo_versao,
    features_utilizadas,
    predicao,
    probabilidade,
    aprovador_humano=None,
):
    """Grava um evento de decisao em formato estruturado e imutavel,
    suportando a linhagem (lineage) e a responsabilizacao (accountability)
    exigidas por LGPD/AI Act."""
    evento = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "id_cliente": id_cliente,
        "modelo": {"nome": modelo_nome, "versao": modelo_versao},
        "features_utilizadas": features_utilizadas,
        "predicao": predicao,
        "probabilidade": probabilidade,
        "revisao_humana": aprovador_humano,  # suporte ao direito de revisao (LGPD Art. 20)
    }
    with open("trilha_auditoria_decisoes.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")
    return evento
```

## Cases e Tendências de Mercado
- A governança de IA deixou de ser tema de conferência para se tornar mercado próprio, com fornecedores, consultorias e plataformas especializadas, crescendo a taxas superiores às de TI em geral.
- Três fatores impulsionam esse crescimento: (1) pressão regulatória (LGPD, AI Act, normas setoriais); (2) aumento do risco operacional de modelos em escala; (3) necessidade de confiança e transparência em soluções de IA.
- Estudos recentes estimam o mercado de governança de IA em algumas centenas de milhões de dólares em 2024, com projeção de chegar a bilhões até o fim da década (CAGR superior a 35% ao ano).
- Empresas em setores regulados (bancos, saúde, telecom, utilities) buscam plataformas com catálogo centralizado de modelos/datasets, trilhas de auditoria, monitoramento de risco de modelo (drift, viés, performance) e "governance as code" integrado a CI/CD.
- **Caso Fairlearn em Modelos de Crédito (Microsoft + EY)**: estudo em que um classificador de crédito treinado de forma tradicional apresentava disparidades significativas entre grupos demográficos (raça, sexo, idade); aplicando Fairlearn, a equipe mitigou vieses via otimização sujeita a restrições de equidade, reduzindo a disparidade sem perder desempenho global de forma significativa — evidenciando que fairness envolve trade-offs entre performance e equidade.
- **Responsible AI nas Big Techs** (Microsoft, Google, IBM): princípios públicos de IA responsável e comitês multidisciplinares de revisão técnica e ética. A Microsoft, por exemplo, tem um Responsible AI Standard cobrindo fairness, confiabilidade/segurança, privacidade, inclusão, transparência e accountability.
- **Tendência: Fairness como padrão de pipeline** — ferramentas como Fairlearn deixam de ser "experimento de pesquisa" e passam a integrar o pipeline padrão de desenvolvimento.
- **Governança como disciplina estratégica (C-level)**: ascensão do papel de Chief Data & Analytics Officer (CDAO), responsável por alinhar estratégia de IA, governança de dados/modelos e resultados de negócio.
- **Regulação em setores críticos**: normas como AI Act e DORA (UE) aumentam a exigência de documentação e auditabilidade ponta a ponta.
- **"Governança como Código"**: políticas de risco e thresholds de fairness codificados em pipelines de CI/CD (jobs que falham se métricas forem violadas, continuous training disparado por drift, regras de acesso versionadas em Git).
- **Integração com CX/CRM**: governança de IA como diferencial competitivo e pilar de confiança de marca em soluções de atendimento e recomendação.
- **Profissionalização da função "AI Model Governance"**: surgimento de profissionais dedicados a padronizar processos de construção, deployment e monitoramento de modelos como domínio próprio, complementar a MLOps e Data Governance.

## Checklist de Estudo
- [ ] Sei explicar o que torna um atributo "sensível" e diferenciar proteção legal/regulatória de risco ético/social
- [ ] Sei descrever como o Fairlearn avalia (MetricFrame, by_group, difference) e mitiga viés (pré-processamento, in-processing, pós-processamento)
- [ ] Sei explicar as quatro funções de um Model Card (transparência/auditoria, gestão de risco, accountability, monitoramento contínuo) e como ele atua como gate de governança em CI/CD
- [ ] Sei diferenciar Model Drift de Data Drift e explicar como o ciclo de feedback contínuo os previne/corrige
- [ ] Sei listar os quatro pilares da governança de modelos (versionamento, rastreabilidade, segurança, responsabilidade/reprodutibilidade)
- [ ] Sei diferenciar as exigências de LGPD e AI Act aplicadas a modelos de ML
- [ ] Sei descrever os papéis (Cientista de Dados, MLOps, Compliance, Gestores, Comitê de Ética) e como o RACI organiza suas responsabilidades

## Palavras-chave
Governança de Modelos. Ciclo de Feedback. MLOps.

## Referências
- ANPD; IAPP. *Insights from the ANPD's new technical note on automated decisions*. 2025. Disponível em: https://iapp.org/news/a/insights-from-the-anpd-s-new-technical-note-on-automated-decisions. Acesso em: 28 jan. 2026.
- AWS – AMAZON WEB SERVICES, INC. *Centralize model governance with SageMaker Model Registry and AWS RAM*. 2024. Disponível em: https://aws.amazon.com/pt/blogs/machine-learning/centralize-model-governance-with-sagemaker-model-registry-resource-access-manager-sharing/. Acesso em: 28 jan. 2026.
- BRAUN, M. *MLOps and Model Governance*. s.d. Disponível em: https://ml-ops.org. Acesso em: 28 jan. 2026.
- DUDÍK, M. et al. *Assessing and mitigating unfairness in credit models with Fairlearn*. 2020. Disponível em: https://www.microsoft.com/en-us/research/wp-content/uploads/2020/09/Fairlearn-EY_WhitePaper-2020-09-22.pdf. Acesso em: 28 jan. 2026.
- IBM. *Preparing for the EU AI Act: Getting governance right*. 2024. Disponível em: https://www.ibm.com/think/insights/eu-ai-act. Acesso em: 28 jan. 2026.
- MARKETSANDMARKETS. *AI Governance Market – Global Forecast to 2029*. 2024. Disponível em: https://www.marketsandmarkets.com/Market-Reports/ai-governance-market-176187291.html. Acesso em: 28 jan. 2026.
- MICROSOFT. *Responsible AI at Microsoft*. s.d. Disponível em: https://www.microsoft.com/en-us/ai/responsible-ai. Acesso em: 28 jan. 2026.
- MITCHELL, M. et al. *Model Cards for Model Reporting*. 2019. Disponível em: https://arxiv.org/abs/1810.03993. Acesso em: 28 jan. 2026.
- MLFLOW DOCUMENTATION. *MLflow: A Tool for Managing the Machine Learning Lifecycle*. s.d. Disponível em: https://mlflow.org. Acesso em: 28 jan. 2026.
- MODELOP. *What is AI Governance? AI Governance Insights from 2024*. 2024. Disponível em: https://modelop.com. Acesso em: 28 jan. 2026.
- NIST. *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*. 2023. Disponível em: https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf. Acesso em: 28 jan. 2026.
- SAI, A. *Retraining Model During Deployment: Continuous Training and Continuous Testing*. 2025. Disponível em: https://neptune.ai/blog/retraining-model-during-deployment-continuous-training-continuous-testing. Acesso em: 28 jan. 2026.
- SDG GROUP. *Databricks Freaky Friday Pills #4: Model Governance*. 2025. Disponível em: https://www.sdggroup.com/en/insights/blog/databricks-freaky-friday-pills-4-model-governance. Acesso em: 28 jan. 2026.
- TECHRADAR PRO. *CDAO responsibilities are evolving: why AI strategy now starts at the top*. 2025. Disponível em: https://www.techradar.com/pro/cdao-responsibilities-are-evolving-why-ai-strategy-now-starts-at-the-top. Acesso em: 28 jan. 2026.
- WEERTS, H. et al. *Fairlearn: Assessing and Improving Fairness of AI Systems*. 2023. Disponível em: https://www.jmlr.org/papers/volume24/23-0389/23-0389.pdf. Acesso em: 28 jan. 2026.
