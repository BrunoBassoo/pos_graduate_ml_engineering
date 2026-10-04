# Aula 04 — Ambientes Reprodutíveis com Docker

## Visão Geral
Aula da disciplina "Engenharia de Software para Cientista de Dados" (Fase 3). Parte do problema clássico de reprodutibilidade — "funciona na minha máquina, mas não na do outro" — causado por diferenças de dependências e sistemas entre ambientes. Apresenta o **Docker** como solução: empacotar aplicações e todas as suas dependências em **contêineres isolados e padronizados**, garantindo o mesmo comportamento em qualquer máquina. A aula combina um hands-on prático de criação de um ambiente Docker para um projeto de análise de dados em Python com um aprofundamento teórico sobre virtualização, gerenciamento de recursos (CPU, memória, GPU), escalabilidade/orquestração de contêineres e boas práticas de engenharia de software aplicadas a ciência de dados.

## Tópicos Abordados
- O problema da reprodutibilidade de ambientes em projetos de ciência de dados
- Hands On: construção de um Dockerfile, build da imagem e execução de um contêiner com persistência de dados via volume
- Contêineres vs. máquinas virtuais (virtualização leve / OS-level virtualization)
- Fundamentos de kernel Linux por trás do Docker: namespaces e cgroups
- Gerenciamento de recursos nos contêineres: CPU, memória e GPU
- Escalabilidade e distribuição de contêineres (Kubernetes, HPA/VPA, balanceamento de carga, Lei de Amdahl)
- Reprodutibilidade científica e distribuição de ambientes (Binder, registries, Docker + Conda)
- Engenharia de software aplicada à ciência de dados com Docker (infraestrutura como código, CI/CD, MLOps)
- Tendências: contêineres distroless, execução rootless, Kubernetes como "SO da nuvem", Singularity/Apptainer em HPC
- Mercado, cases e tendências (Netflix, Uber, Unicamp, IDC)

## Conceitos-Chave

### Contêineres vs. Máquinas Virtuais
VMs tradicionais usam um hipervisor para virtualizar o hardware inteiro, rodando um sistema operacional completo (kernel + user space) por cima — o que consome muito recurso e leva minutos para iniciar. Um contêiner Docker faz **virtualização a nível de sistema operacional (OS-level virtualization)**: compartilha o mesmo kernel do host, mas isola processos, sistema de arquivos e rede através de primitivas do próprio kernel Linux. Na prática, um contêiner é um processo isolado com sua própria visão do sistema (root filesystem próprio, PID 1 próprio, interface de rede própria), por isso é muito mais leve e sobe em segundos.

### Namespaces e Cgroups
O Docker depende de duas funcionalidades-chave do kernel Linux:
- **Namespaces** — implementam o isolamento (PID namespace isola a árvore de processos, NET namespace isola a rede, MNT namespace isola o sistema de arquivos montado, etc.).
- **Cgroups (control groups)** — cuidam do gerenciamento de recursos, permitindo impor limites e quotas de CPU, memória e I/O para grupos de processos (ex.: `--memory`, `--cpus` no `docker run`). Se o contêiner exceder o limite de memória, o kernel pode disparar um **OOM kill**, protegendo o host sem derrubar o sistema.

A padronização desses formatos vem da especificação **OCI (Open Container Initiative)**, que também permite alternativas como o **Podman** (contêineres rootless, sem daemon).

### Gerenciamento de Recursos: CPU, Memória e GPU
Por padrão, um contêiner pode usar todo o recurso disponível no host. O Docker permite restringir isso via cgroups (`docker run --memory=4g --cpus=2`), essencial em cenários multiusuário (vários experimentos rodando em paralelo no mesmo servidor). Para **GPU**, é necessário instalar o NVIDIA Container Toolkit no host e usar a flag `--gpus` (ex.: `docker run --gpus all ... nvidia/cuda:12.2-python3.10`), permitindo treinar modelos de Deep Learning sem perda relevante de desempenho (overhead de containerização estimado em 2-5%, próximo de zero para CPU/I/O/memória). Internamente, o Docker ainda usa **Union File Systems** para as camadas de imagem (cada instrução do Dockerfile gera uma camada somente-leitura, com uma camada de leitura-escrita no topo do contêiner ativo) — por isso, para escrita intensiva de dados (ex.: checkpoints de modelo), recomenda-se usar **volumes montados**.

### Escalabilidade e Orquestração
Contêineres são facilmente replicáveis (cada instância carrega todas as dependências), o que viabiliza escalar horizontalmente um serviço de ML atrás de um balanceador de carga. O **Kubernetes (K8s)** é a plataforma mais adotada para gerenciar contêineres em cluster: define-se um *deployment* com um número de réplicas desejado, e o K8s agenda os *pods* nos nós disponíveis. O **Horizontal Pod Autoscaler (HPA)** ajusta o número de réplicas conforme métricas de uso (ex.: CPU > 80%), enquanto o **Vertical Pod Autoscaler (VPA)** ajusta os recursos de cada contêiner individualmente. A escala horizontal, porém, tem limite teórico dado pela **Lei de Amdahl**: se uma fração P do workload é paralelizável, escalar para N instâncias reduz o tempo dessa parte em até N vezes, mas a parcela sequencial (1-P) permanece como gargalo — por isso, às vezes melhorar o algoritmo vale mais do que apenas adicionar contêineres.

### Reprodutibilidade Científica e Distribuição de Ambientes
Contêineres publicados em registries (Docker Hub ou privados) permitem compartilhar ambientes complexos integralmente. O projeto **Binder** (Jupyter) gera automaticamente uma imagem Docker a partir de um repositório GitHub com Dockerfile/requirements, permitindo executar notebooks sem instalação local. Conferências como a **NeurIPS** incentivam autores a publicarem contêineres junto com o código de seus artigos, reforçando a reprodutibilidade científica. Combinando **Docker + Conda**, um estudo citado no material aponta ~99% de reprodutibilidade de ambientes, muito superior ao uso isolado de virtualenvs/pip, pois elimina variâncias de sistema operacional e resolve conflitos de bibliotecas nativas.

### Engenharia de Software e MLOps com Docker
Tratar o ambiente de execução como código (Dockerfile versionado no Git) traz práticas maduras de DevOps para a ciência de dados: mudanças de ambiente tornam-se rastreáveis e reversíveis, e pipelines de **CI/CD** podem construir a imagem e rodar testes a cada commit, seguindo o princípio de **infraestrutura imutável**. Imagens Docker podem ser compostas/estendidas (imagem base padronizada da equipe + camadas específicas do projeto), promovendo reuso. Em **MLOps**, ferramentas como **MLflow** podem exportar um modelo registrado diretamente como imagem Docker, e o **Kubeflow** estrutura pipelines em que cada etapa é um contêiner — o deploy de um modelo em produção passa a ser, essencialmente, a entrega de um contêiner.

## Exercício Hands-On (do material)
O hands-on acompanha videoaulas mostrando a criação de um ambiente Docker para um projeto simples de análise de dados em Python (pandas e numpy), cobrindo o fluxo completo:

1. Escrever um **Dockerfile** definindo imagem base, diretório de trabalho, cópia de arquivos, instalação de dependências e comando padrão.
2. **Construir a imagem** com `docker build`.
3. **Executar o contêiner** com persistência de dados via volume montado.
4. Boas práticas destacadas: manter imagens leves e seguras, fixar versões de dependências e nunca inserir informações sensíveis no Dockerfile.

## Exemplos de Código

Dockerfile para um projeto de análise de dados em Python:

```dockerfile
# Imagem base enxuta com Python 3.9
FROM python:3.9-slim

# Diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia primeiro as dependências para aproveitar cache de camadas
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copia o restante do código do projeto
COPY . .

# Comando padrão executado ao iniciar o contêiner
CMD ["python", "analise.py"]
```

Build da imagem e execução do contêiner com persistência de dados:

```bash
# Constrói a imagem a partir do Dockerfile no diretório atual
docker build -t meu-projeto:1.0 .

# Executa o contêiner, nomeando-o e montando um volume local
# para persistir os dados gerados/consumidos pela análise
docker run -it --name analise-container \
  -v $(pwd)/dados:/app/dados \
  meu-projeto:1.0
```

Controle fino de recursos (CPU/memória) e uso de GPU:

```bash
# Limita o contêiner a 4 GB de RAM e ~2 CPUs
docker run --memory=4g --cpus=2 meu-projeto:1.0

# Habilita acesso à GPU NVIDIA do host (requer NVIDIA Container Toolkit)
docker run --gpus all nvidia/cuda:12.2-python3.10 python treino.py
```

## Cases e Tendências de Mercado
Segundo a IDC, cerca de **65% das empresas da Fortune 500** já utilizam ambientes conteinerizados para projetos de IA/ML em 2025, evidenciando a maturidade da tecnologia no mercado corporativo. Plataformas como **Binder** (Jupyter) e **MLflow** integram Docker para garantir reprodutibilidade e facilitar o deploy de modelos.

**Cases ilustrativos:**
- **Netflix** utiliza contêineres para orquestrar notebooks Jupyter e pipelines de recomendação em escala global, com Kubernetes para escalar jobs de treinamento e tunagem de algoritmos.
- **Uber** emprega Docker no sistema **Michelangelo** (plataforma de ML end-to-end) para isolar jobs de treinamento e servir modelos em múltiplos datacenters.
- **Unicamp** publicou a série "Engenharia e Ciência de Dados Cidadã 2025", com aulas práticas sobre Docker e PostgreSQL aplicados a dados reais.

**Publicações e vídeos abertos citados no material:**
- "Docker para Ciência de Dados – Guia Passo a Passo"
- "Descomplicando o Docker" – Marcelo Rodrigues, DIO
- "Docker + IA: Acabei com o 'funciona na minha máquina'" – Sergio Monteiro
- "Docker e Reprodutibilidade" – Workshop NeurIPS

**Tendências apontadas:** imagens **distroless** (sem shell/gerenciador de pacotes, menor superfície de ataque); contêineres **rootless** (Podman, Docker rootless mode) para maior segurança em ambientes multi-tenant; consolidação do **Kubernetes** como "sistema operacional da nuvem"; uso de **Mamba** para acelerar builds com Conda; e adoção de **Singularity/Apptainer** em HPC e pesquisa em IA, compatíveis com imagens Docker mas sem exigir privilégios de superusuário.

## Checklist de Estudo
- [ ] Sei explicar por que o Docker resolve o problema de "funciona na minha máquina"
- [ ] Sei escrever um Dockerfile básico (FROM, WORKDIR, COPY, RUN, CMD) para um projeto Python
- [ ] Sei construir uma imagem (`docker build`) e executar um contêiner com volume persistente (`docker run -v`)
- [ ] Sei diferenciar contêiner de máquina virtual em termos de virtualização (OS-level vs. hipervisor)
- [ ] Sei explicar o papel de namespaces (isolamento) e cgroups (controle de recursos) no Docker
- [ ] Sei limitar CPU/memória de um contêiner e habilitar acesso a GPU via NVIDIA Container Toolkit
- [ ] Sei explicar como Kubernetes escala contêineres horizontalmente (deployment, HPA, VPA) e a relação com a Lei de Amdahl
- [ ] Sei descrever como Docker + Conda melhoram a reprodutibilidade de ambientes de ciência de dados
- [ ] Sei relacionar Docker com práticas de infraestrutura como código, CI/CD e MLOps
- [ ] Conheço cases de mercado (Netflix, Uber/Michelangelo, Unicamp) e tendências (distroless, rootless, Singularity/Apptainer)

## Palavras-chave
Reprodutibilidade computacional. Conteinerização. Engenharia de software para ciência de dados.

## Referências
- CHOUDHARY, A. **DevOps for Data Science**: Reproducible Environments with Docker and Conda. 2025. Disponível em: https://johal.in/devops-for-data-science-reproducible-environments-with-docker-and-conda-2025/. Acesso em: 15 dez. 2025.
- MONTEIRO, S. A. **Docker + IA**: Acabei com o "funciona na minha máquina". 2025. Disponível em: https://pt.linkedin.com/pulse/docker-ia-acabei-com-o-funciona-na-minha-m%C3%A1quina-assun%C3%A7%C3%A3o-monteiro-sipwf. Acesso em: 15 dez. 2025.
- NICKOLOFF, J.; KUENZLI, S. **Docker in Action** – Second Edition. Shelter Island: Manning Publications, 2019.
- RODRIGUES, M. **Descomplicando o Docker**: Como Contêineres Estão Revolucionando o Deploy. 2025. Disponível em: https://www.dio.me/articles/descomplicando-o-docker-como-conteineres-estao-revolucionando-o-deploy-1c7b7cf19599. Acesso em: 15 dez. 2025.
- VICTORINO, V. **Docker**: "funciona na minha máquina". 2022. Disponível em: https://viniciusflv.github.io/pt-BR/posts/devops/docker/. Acesso em: 15 dez. 2025.
