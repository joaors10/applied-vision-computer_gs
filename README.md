# OrbitalWatch Vision

Classificação de imagens do Sentinel-2 (EuroSAT) para o projeto OrbitalWatch — Global Solution, eixo Indústria Espacial.

## Integrantes

| Nome | RM |
|------|-----|
| João Rodrigo Solano Nogueira | 551319 |
| Julia Amorim Bezerra | 99609 |
| Lana Giulia Auada Leite | 551143 |
| Tony Willian da Silva Segalin | 550667 |

## Objetivo

Classificar recortes RGB em 10 classes de cobertura/uso do solo (floresta, área urbana, agricultura, rio, etc.) com duas CNNs treinadas do zero, sem modelos pré-treinados.

## Dataset

- Fonte: [EuroSAT](https://github.com/phelber/EuroSAT)
- 10 classes, ~27 mil imagens
- Divisão: 70% treino, 15% validação, 15% teste (`python eurosat.py`, seed 42)
- Imagens 64×64, normalização 1/255, flip horizontal no treino

Estrutura após o split:

```
dataset_eurosat/
  2750/
  train/
  val/
  test/
```

Se o repositório não incluir as pastas `train/`, `val/` e `test/` (tamanho), baixe o EuroSAT, extraia em `2750/` e execute `python eurosat.py`.

## Ambientes do projeto

O projeto possui duas formas de execução:

1. **Treinamento e análise (ML)**
   - Notebook Jupyter
   - Modelos CNN

2. **Demonstração funcional (Streamlit)**
   - Interface web para predição com imagens novas

## Instalação

## Instalação (Treinamento / Notebook)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```


---

## 3. Criando ambiente de teste separado.

```md
## Demonstração funcional (Streamlit)

Para rodar a interface de predição:

```bash
python3.12 -m venv .venv_app
source .venv_app/bin/activate
pip install streamlit
streamlit run app.py
```

`TRAIN_MODELS = False` usa os pesos em `models/` e os gráficos em `outputs/`.  
`TRAIN_MODELS = True` treina as duas redes no notebook.

## Treino (script)

```bash
python src/train.py --data-dir dataset_eurosat --model simple_cnn --epochs 25 --image-size 64
python src/train.py --data-dir dataset_eurosat --model deep_cnn --epochs 25 --image-size 64
```

Saídas: `models/` (pesos) e `outputs/` (curvas, matriz de confusão, métricas JSON).

## Predição

```bash
python src/predict.py \
  --model-path models/deep_cnn_best.keras \
  --classes-path models/deep_cnn_classes.json \
  --image-path caminho/imagem.jpg \
  --image-size 64
```

## Demo Streamlit

```bash
streamlit run app.py
```

## Arquivos principais

| Arquivo | Função |
|---------|--------|
| `notebook.ipynb` | Treino, avaliação e relatório |
| `src/models.py` | Arquiteturas CNN |
| `src/train.py` | Treino por linha de comando |
| `src/data.py` | Leitura do dataset |
| `src/predict.py` | Classificar uma imagem |
| `eurosat.py` | Split train/val/test |
| `app.py` | Interface de teste |

## Resultados (teste)

| Modelo | Acurácia | Loss |
|--------|----------|------|
| Simple CNN | ~84% | ~0,47 |
| Deep CNN | ~90,7% | ~0,28 |

Meta do trabalho: ≥ 88% — atingida pelo Deep CNN.
