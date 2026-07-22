# Processo Seletivo – Intensivo Maker | AI

**Candidato:** Matheus Vinicius Vidal de Andrade  
**Projeto escolhido:** Projeto 1 — Classificação MNIST

---

## 📌 Visão Geral

Classificação de dígitos manuscritos (0-9) usando **MNIST dataset** com CNN treinada do zero, convertida para **TensorFlow Lite** com quantização dinâmica para Edge AI.

Pipeline completo: `treinamento → validação → salvamento → conversão TFLite → inferência`

---

## 📂 Estrutura do Repositório

```
projetos/1-classificacao-mnist/          ← Projeto escolhido
├── train_model.py                        Treinamento da CNN
├── optimize_model.py                     Conversão para TFLite com otimização
├── run_inference.py                      Inferência com model.tflite
├── requirements.txt                      tensorflow>=2.12, numpy
├── model.h5                              Modelo treinado (~1.3 MB)
├── model.tflite                          Modelo otimizado para edge (~114 KB)
└── README.md                             Instruções + relatório do candidato
.github/
├── workflows/ci.yml                       Validação automática (GitHub Actions)
└── scripts/                               Scripts de validação (não modificar)
```

---

## ⚡ Como Executar Localmente

### 1. Instalar dependências

```bash
cd projetos/1-classificacao-mnist
pip install -r requirements.txt
```

### 2. Treinar o modelo

```bash
python train_model.py       # gera model.h5 (+ imprime acurácia de validação)
```

### 3. Otimizar para TFLite

```bash
python optimize_model.py    # gera model.tflite (+ imprime redução de tamanho)
```

### 4. Rodar inferência de exemplo

```bash
python run_inference.py     # carrega model.tflite e testa 5 amostras
```

---

## 🤖 GitHub Actions (CI)

O workflow em `.github/workflows/ci.yml` é acionado automaticamente em todo `push` ou `pull_request` para `main`.

**O que a CI faz:**

1. Detecta qual projeto está em `projetos/` (deve restar apenas 1 pasta)
2. Verifica se todos os arquivos obrigatórios existem (`model.h5`, `model.tflite`, `train_model.py`, etc.)
3. Valida a qualidade dos artefatos:
   - **model.h5:** acurácia ≥ 85% em 300 amostras de teste
   - **model.tflite:** acurácia ≥ 75% em 300 amostras de teste
4. Reexecuta `train_model.py`, `optimize_model.py` e `run_inference.py` — todos devem rodar sem erro
5. Verifica se o TFLite é menor que o `.h5` (quantização aplicada)

**Para ver o resultado:**
- Acesse a aba **Actions** do seu repositório no GitHub
- Clique no workflow mais recente
- Expandir cada step para ver logs detalhados

> ⚠️ A CI não treina nem otimiza por você. Os artefatos (`model.h5`, `model.tflite`) **devem estar commitados** no repositório — a CI apenas os valida.

---

## ⚠️ Restrições de Engenharia

- **CPU-only** (`device="cpu"`)
- Sem modelos pré-treinados
- Máximo 15 épocas (com EarlyStopping)
- Entrada: imagens `(28, 28, 1)` normalizadas em [0, 1]
- Arquivos gerados localmente e commitados

---

## ✅ Critérios de Avaliação (MNIST)

| Critério | Detalhes |
|----------|----------|
| Treinamento | 3 blocos Conv2D+BatchNorm+MaxPooling, Dropout, EarlyStopping, split validação |
| Métrica | Acurácia de validação ≥ 85% (model.h5) e ≥ 75% (model.tflite) |
| Otimização | Dynamic Range Quantization (tf.lite.Optimize.DEFAULT) |
| Inferência | 5+ amostras com model.tflite, saída predito vs. real |
| Documentação | Relatório de 6 seções preenchido no README do projeto |

---

## 📤 Como Enviar (Push para o GitHub)

```bash
git add .
git commit -m "Entrega desafio - Matheus Vinicius Vidal de Andrade"
git push origin main
```

Após o push, a CI roda automaticamente. Acompanhe em **Actions**.
****
