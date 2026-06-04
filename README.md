# OrbitalWatch Vision

Classificação de imagens do Sentinel-2 (EuroSAT) usando redes neurais convolucionais treinadas do zero.

## Integrantes

| Nome | RM |
|------|-----|
| João Rodrigo Solano Nogueira | 551319 |
| Julia Amorim Bezerra | 99609 |
| Lana Giulia Auada Leite | 551143 |
| Tony Willian da Silva Segalin | 550667 |

## Objetivo

Classificar imagens RGB em 10 classes de uso do solo (floresta, urbano, agricultura, rios, etc.) utilizando CNNs desenvolvidas do zero.

## Dataset

EuroSAT (Sentinel-2), com aproximadamente 27 mil imagens distribuídas em 10 classes. As imagens possuem tamanho 64x64 RGB e foram normalizadas com 1/255. O dataset foi dividido em 70% treino, 15% validação e 15% teste com seed 42.

Estrutura esperada:

dataset_eurosat/
  2750/
  train/
  val/
  test/

Caso as pastas de treino, validação e teste não existam, execute:

python eurosat.py

## Instalação

python3.12 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

## Treinamento

O treinamento pode ser feito via notebook ou scripts.

Notebook:

jupyter notebook notebook.ipynb

Scripts:

python src/train.py --data-dir dataset_eurosat --model simple_cnn --epochs 25 --image-size 64  
python src/train.py --data-dir dataset_eurosat --model deep_cnn --epochs 25 --image-size 64  

## Predição

python src/predict.py \
  --model-path models/deep_cnn_best.keras \
  --classes-path models/deep_cnn_classes.json \
  --image-path caminho/imagem.jpg \
  --image-size 64  

## Demonstração funcional (Streamlit)

python3.12 -m venv .venv_app  
source .venv_app/bin/activate  
pip install streamlit  
streamlit run app.py  

## Resultados

Simple CNN: ~84% acurácia | ~0.47 loss  
Deep CNN: ~90.7% acurácia | ~0.28 loss  

Meta do projeto: ≥ 88% (atingida pelo modelo Deep CNN)

## Arquivos principais

notebook.ipynb → treino e avaliação  
src/models.py → arquiteturas CNN  
src/train.py → treinamento  
src/data.py → carregamento do dataset  
src/predict.py → inferência  
eurosat.py → split do dataset  
app.py → interface Streamlit
