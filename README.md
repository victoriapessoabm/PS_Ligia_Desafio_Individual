# 🩺 Desafio Lígia – Classificação de Pneumonia em Raios-X

Este repositório contém todo o fluxo de desenvolvimento da solução de classificação binária de pneumonia em radiografias, incluindo análise exploratória, preparação dos dados, modelagem, interpretabilidade e geração da submissão final. Todo o projeto foi configurado para rodar localmente, usando apenas caminhos relativos e mantendo portabilidade independente da máquina utilizada.

---

## 📁 Estrutura do Repositório

```text
PS_Ligia_Desafio_Individual/
├── BestModel/
│   └── best_model.keras
├── data/
│   ├── chest_xray/
│   │   ├── train/
│   │   └── test/
│   └── ligia-compviz/
│       ├── test.csv
│       └── test_images/test_images/
├── ImagePreprocessing/
│   └── imagePipeline.py
├── Interpretability/
│   ├── saliency.py
│   └── lime.py
├── Notebooks/
│   ├── EDA.ipynb
│   ├── Modelagem.ipynb
│   └── Inferencia.ipynb
├── Submission/
├── src/
│   └── generateSubmission.py
└── requirements.txt

# 📓 Notebooks incluídos

EDA.ipynb — análise exploratória, visualização das imagens e preparação do dataset.

Modelagem.ipynb — construção dos modelos, avaliação, validação e escolha do modelo final.

Inferencia.ipynb — inferência local, métricas finais e técnicas de interpretabilidade (Grad-CAM, Saliency e LIME).

⚠️ O notebook Modelagem não roda localmente devido ao alto custo computacional.
Ele serve como documentação completa do processo de treinamento.
