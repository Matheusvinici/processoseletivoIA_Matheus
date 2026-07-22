# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Matheus Vinicius Vidal de Andrade

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada possui 3 blocos convolucionais sequenciais, cada um composto por:
- **Conv2D** com filtros 32, 64 e 128, kernel 3×3, padding "same" e ativação ReLU
- **BatchNormalization** para estabilizar e acelerar o treinamento
- **MaxPooling2D** com pool 2×2 para redução de dimensionalidade

Após os blocos convolucionais, aplica-se **Flatten**, seguido de **Dropout(0.3)** como regularização antes da camada de saída **Dense(10, softmax)**.

A validação é feita com split manual: os últimos 5.000 exemplos do conjunto de treino (60k) são reservados para validação. O treinamento utiliza **EarlyStopping** monitorando `val_loss` com `patience=3` e `restore_best_weights=True`, limitado a no máximo 15 épocas.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow** 2.21.0
- **NumPy** 2.5.1
- **Keras** 3.15.0

### 3️⃣ Técnica de Otimização do Modelo

Foi aplicada **Dynamic Range Quantization** via `converter.optimizations = [tf.lite.Optimize.DEFAULT]`. Esta técnica reduz a precisão dos pesos de float32 para int8 durante a serialização, mantendo as ativações em float32 na inferência. O resultado é uma redução significativa no tamanho do modelo (~4×) com perda mínima de acurácia, ideal para dispositivos Edge.

### 4️⃣ Resultados Obtidos

- **Acurácia de validação:** 99,18%
- **model.h5:** 1294,9 KB
- **model.tflite:** 114,0 KB
- **Redução:** 91,2%

### 5️⃣ Comentários Adicionais (Opcional)

Treinamento em CPU apenas, como solicitado. O EarlyStopping interrompeu o treinamento antes das 15 épocas máximas (provavelmente entre épocas 6-9), restaurando os melhores pesos. A quantização dinâmica reduziu o modelo para ~9% do tamanho original, tornando-o adequado para implantação em dispositivos Edge como microcontroladores e smartphones.

### 6️⃣ Exemplo de Inferência

```
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

Todas as 5 amostras foram classificadas corretamente, demonstrando que a quantização dinâmica não comprometeu a capacidade preditiva do modelo neste conjunto de teste. Como as amostras iniciais do MNIST (dígitos 7, 2, 1, 0, 4) são razoavelmente nítidas, era esperado que o modelo acertasse todas — em testes com amostras mais desafiadoras ou ambíguas, a taxa de acerto poderia ser ligeiramente inferior à do modelo float32 original.
