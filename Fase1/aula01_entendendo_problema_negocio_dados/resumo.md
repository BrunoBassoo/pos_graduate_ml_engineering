# Aula 01 — Entendendo o Problema de Negócio e os Dados

## Visão Geral
Aula introdutória da Fase 1 (ML Engineering). Defende que a etapa mais crítica de um projeto de ML não é a modelagem, e sim o **entendimento profundo do problema de negócio e dos dados** — equipes de sucesso dedicam 20-30% do cronograma a isso. A metodologia central é o **CRISP-DM**, com foco na fase de **Business Understanding**.

## Tópicos Abordados
- Por que projetos de IA falham (desalinhamento com o negócio, não limitação técnica)
- Hands On: tradução de um pedido vago de negócio em projeto de ML mensurável (caso FIAPMobile / churn)
- CRISP-DM — fase de Business Understanding e suas 4 tarefas
- Como levantar requisitos, restrições, pressupostos e expectativas
- Identificação de variáveis (features) e KPIs, e a variável-alvo (target)
- Envolvimento de stakeholders (quem envolver e como)
- Exemplos aplicados: Saúde (readmissão hospitalar) e Varejo (previsão de demanda)
- Mercado, cases e tendências (taxas de falha, CRISP-ML(Q), Data Product Managers)

## Conceitos-Chave

### CRISP-DM — Business Understanding (4 tarefas)
1. **Determinar os objetivos do negócio** — entender o que o cliente quer alcançar + critério de sucesso do negócio.
2. **Avaliar a situação** — inventário de recursos (pessoas, dados, infraestrutura, tempo), requisitos/premissas/restrições, riscos e contingências, glossário, análise de custo-benefício.
3. **Determinar os objetivos de data science** — traduzir o objetivo de negócio em objetivo técnico mensurável (ex.: acurácia mínima, AUC).
4. **Elaborar o plano de projeto** — cronograma, responsáveis, dependências, iterações, ferramentas.

### Requisitos, Restrições, Pressupostos e Expectativas
- **Requisitos**: prazos, performance mínima, interpretabilidade, compliance/LGPD.
- **Restrições**: dados limitados, orçamento, tecnologia legada, prazo fixo, equipe reduzida.
- **Pressupostos**: coisas assumidas sem confirmação (ex.: "dado X estará disponível") — devem ser validadas.
- **Expectativas dos stakeholders**: o que cada parte espera receber (dashboard, lista priorizada, API etc.).

### KPIs vs. Métricas Técnicas
Um projeto bem alinhado tem uma cadeia clara: **métrica técnica → impacto no KPI de negócio**.
Exemplo: *"Acurácia do modelo de 90% (métrica técnica) permitirá interceptar 30% mais fraudes, resultando em redução de 20% nas perdas (KPI de negócio)"*.

### Variável-alvo (Target)
Definir claramente o que é o evento a prever (ex.: churn = cancelamento efetivo ou também inatividade? readmissão em 30 ou 60 dias?) — decisão de negócio, não técnica, que define o tipo de tarefa (classificação, regressão, clusterização).

### Stakeholders típicos
Sponsor executivo · Especialistas de domínio · Donos dos dados (data owners) · Usuários finais/operacionais · Parceiros/fornecedores.

## Exercício Hands-On (do material)
**Caso FIAPMobile** (operadora fictícia com alto churn). Atividade proposta — responder:
1. Definição do problema de negócio (ex.: "Reduzir em 15% a taxa de churn trimestral").
2. Métrica de sucesso / KPI (ex.: aumento da taxa de retenção).
3. Perguntas para levantar requisitos e restrições junto aos stakeholders (prazos, orçamento, LGPD).
4. Dados e variáveis relevantes (tempo de uso, reclamações, pagamentos em atraso, dados externos).
5. Quais stakeholders envolver e como (CRM, suporte, marketing, equipe de retenção).

> Este exercício não envolve código — é um exercício de formulação de problema e comunicação com stakeholders, propositalmente anterior a qualquer modelagem.

## Exemplo de Código — Formalizando um "Project Charter" de ML

Embora esta aula seja conceitual, o material que sai do Business Understanding pode (e deve) ser versionado como artefato do projeto. Um padrão útil em MLOps é registrar isso em um arquivo YAML que acompanha o repositório do projeto:

```yaml
# project_charter.yaml — artefato produzido ao final do Business Understanding
projeto: "Redução de Churn - FIAPMobile"
objetivo_negocio: >
  Reduzir em 15% a taxa de churn trimestral, identificando clientes em risco
  de cancelamento com antecedência.
kpi_negocio:
  nome: "Taxa de churn trimestral"
  baseline: 0.22
  meta: 0.187          # -15% relativo
objetivo_data_science:
  tarefa: "classificacao_binaria"
  target: "churned_90d"       # churn = 90 dias de inatividade (definição de negócio)
  metrica_tecnica: "recall"
  meta_tecnica: 0.85
restricoes:
  - "Dados de clientes sujeitos à LGPD — anonimização obrigatória"
  - "Modelo deve ser interpretável (requisito da diretoria)"
  - "Prazo: 3 meses para entrar na próxima campanha de retenção"
stakeholders:
  sponsor: "Diretoria de Marketing"
  especialistas_dominio: ["Time de CRM", "Suporte Técnico"]
  donos_dados: ["TI / Data Warehouse"]
  usuarios_finais: ["Equipe de Retenção de Clientes"]
variaveis_candidatas:
  - tempo_de_contrato_meses
  - numero_chamadas_suporte_90d
  - valor_fatura_media
  - atraso_pagamento_dias
  - uso_dados_mb_tendencia
```

Esse arquivo funciona como o "contrato social" descrito na aula, versionável no Git, e serve de base auditável para as próximas fases (Data Understanding, Modelagem).

## Cases e Tendências de Mercado
- Apenas **~13%** dos projetos de Data Science chegam à produção (VentureBeat, 2019).
- Gartner: até 2022, só **20%** das iniciativas analíticas entregaram resultado de negócio.
- **38%** das causas de fracasso citam "compreensão do negócio"; **24%** citam "compreensão dos dados" (pesquisa 2021).
- **CRISP-ML(Q)** (2021): funde Business Understanding + Data Understanding em uma fase única, com foco em qualidade.
- Tendência de **Data Product Managers** e processos de *discovery*/design thinking antes de codar.
- Cases: Netflix (foco em product metrics antes de modelar), Walmart (meses de entendimento do problema logístico antes da roteirização).

## Checklist de Estudo
- [ ] Sei explicar as 4 tarefas do Business Understanding do CRISP-DM
- [ ] Sei diferenciar requisito, restrição, pressuposto e expectativa
- [ ] Sei distinguir métrica técnica de KPI de negócio e dar um exemplo da cadeia entre elas
- [ ] Sei listar pelo menos 4 tipos de stakeholders e o papel de cada um
- [ ] Consigo aplicar o framework nos dois exemplos (saúde e varejo) e explicar o que daria errado sem o alinhamento inicial

## Palavras-chave
CRISP-DM · Business Understanding · Ciclo de vida de modelos · KPI · Stakeholders

## Referências
- WIRTH, R.; HIPP, J. *CRISP-DM: Towards a Standard Process Model for Data Mining*. 2000.
- STUDER, S. et al. *Towards CRISP-ML(Q)*. 2021. https://doi.org/10.3390/make3020020
- PROVOST, F.; FAWCETT, T. *Data Science for Business*. O'Reilly, 2013.
- DATA SCIENCE PM. *CRISP-DM 2.0 – The Business Understanding Phase*. 2023.
- JOSHI, M. P. et al. *Why So Many Data Science Projects Fail to Deliver*. MIT Sloan, 2021.
