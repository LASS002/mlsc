# MLSC: Machine Learning Square vs Circle

Bem-vindo ao projeto **MLSC** (Machine Learning Square vs Circle). Este repositório contém um pipeline completo e didático para criar um modelo de aprendizado de máquina capaz de distinguir entre imagens de quadrados e círculos.

Este projeto foi desenhado para ser um recurso educacional, cobrindo desde a geração de dados sintéticos até o treinamento de uma Rede Neural Convolucional (CNN) usando PyTorch.

## 🎯 Objetivo

O objetivo principal é demonstrar, passo a passo, como construir um classificador de imagens simples, porém funcional. Você aprenderá sobre:

- **Geração de Dados Sintéticos**: Como criar seu próprio dataset usando Python.
- **Processamento de Dados**: Como carregar e preparar dados para treinamento.
- **Deep Learning**: Conceitos fundamentais de CNNs.
- **Engenharia de MLOps**: Estrutura de projeto limpa e gerenciamento de dependências com `uv`.

## 📂 Estrutura do Projeto

O projeto segue uma estrutura plana ("flat layout") para simplicidade e clareza.

```text
mlsc/ (toda a estrutura do projeto)
├── pyproject.toml      # Gerenciamento de dependências, configuração do projeto e entry point
├── README.md           # Este arquivo (Documentação Inicial)
├── uv.lock             # Arquivo de bloqueio de versões (garantia de reprodutibilidade)
├── data/               # Diretório onde os dados residem
│   ├── raw/            # Dados brutos gerados (imagens .png)
│   │   ├── square/     # Imagens de quadrados
│   │   └── circle/     # Imagens de círculos
├── mlsc/               # Código fonte do pacote
│   ├── __init__.py     # Torna o diretório um pacote Python
│   ├── mlsc.py         # Ponto de entrada principal (CLI)
│   ├── generate_data.py # Script para gerar as imagens sintéticas
│   ├── dataset.py      # Definição da classe Dataset (carregamento de dados)
│   ├── model.py        # Arquitetura da Rede Neural (SimpleCNN)
│   └── train.py        # Loop de treinamento e validação
└── docs/               # Documentação complementar
    ├── help.md         # Documentação Técnica e Acadêmica (Nível Ph.D.)
    └── help.html       # Versão HTML da documentação técnica
```

## 🚀 Como Executar

Este projeto utiliza o **uv** para gerenciamento de dependências, que é uma ferramenta extremamente rápida e moderna para Python.

### 1. Instalação

Primeiro, certifique-se de ter o `uv` instalado. Se não tiver, consulte a [documentação oficial do uv](https://github.com/astral-sh/uv).

Em seguida, instale as dependências do projeto e o próprio pacote em modo editável:

```bash
uv sync
```

### 2. Geração de Dados

Utilize o comando `mlsc` via `uv run` para gerar o dataset:

```bash
uv run mlsc generate
```

*O que isso faz?* Cria 2000 imagens (1000 quadrados, 1000 círculos) de 64x64 pixels e as salva em `data/raw`.

### 3. Treinamento do Modelo

Treine a Rede Neural utilizando o subcomando `train`:

```bash
uv run mlsc train
```

*O que isso faz?*

1. Carrega as imagens geradas.
2. Divide em treino (80%) e validação (20%).
3. Treina a `SimpleCNN` por 10 épocas.
4. Exibe a perda (loss) e acurácia a cada época.
5. Salva o modelo treinado em `model.pth`.

## 🧠 Entendendo o Modelo (SimpleCNN)

Utilizamos uma **Rede Neural Convolucional (CNN)**, que é o padrão ouro para processamento de imagens.

1. **Camadas Convolucionais (`Conv2d`)**: Funcionam como filtros que aprendem a identificar características visuais (bordas, cantos, curvas).
2. **Função de Ativação (`ReLU`)**: Introduz não-linearidade, permitindo que a rede aprenda padrões complexos.
3. **Pooling (`MaxPool2d`)**: Reduz a dimensão espacial da imagem, tornando o processamento mais eficiente e o modelo mais robusto a pequenas variações de posição.
4. **Camada Linear (`Linear`)**: Toma as características extraídas e faz a classificação final (Quadrado ou Círculo).

## 📚 Documentação Avançada

Para uma compreensão teórica profunda, incluindo formulações matemáticas de backpropagation e detalhes arquiteturais, consulte a documentação acadêmica em `docs/help.md` ou abra `docs/help.html` em seu navegador.

---
*Lennin Abrão Sousa Santos. Projeto desenvolvido para fins educacionais.*
