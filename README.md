🩺 Desafio Lígia – Classificação de Pneumonia em Raios-X

Este repositório contém todo o fluxo de desenvolvimento de uma solução de classificação binária de pneumonia em radiografias, incluindo análise exploratória, preparação dos dados, modelagem, interpretabilidade, inferência e geração da submissão final.
Todo o projeto foi configurado para rodar localmente, usando apenas caminhos relativos e mantendo portabilidade independente da máquina utilizada.

🧬 Clonar o Repositório

    git clone https://github.com/victoriapessoabm/PS_Ligia_Desafio_Individual.git
    cd PS_Ligia_Desafio_Individual
    


📁 Estrutura do Repositório

    PS_Ligia_Desafio_Individual/
    ├── BestModel/
    │   └── best_model.keras
    ├── data/
    │   ├── chest_xray/
    │   └── ligia-compviz/
    ├── ImagePreprocessing/
    │   └── imagePipeline.py
        └── modelBuilder.py
        └── preprocessing.py
        └── generate_csv.py
    ├── Interpretability/
    │   ├── saliency.py
    │   └── lime.py
    ├── Notebooks/
    │   ├── EDA.ipynb
    │   ├── Modelagem.ipynb
    │   └── Inferencia.ipynb
    ├── Submission/
        └── submission.csv
    ├── src/
    │   └── generateSubmission.py
    └── requirements.txt

📥 Como Obter os Dados e Preparar o Diretório data/
1. Dataset de Raios-X Rotulados (treino / validação / interpretação)

  1. Acessar o dataset no Kaggle:
      Labeled Chest X-Ray Images: https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
  2. Clicar em Download (será baixado um arquivo archive.zip).
  3. Descompactar o archive.zip.
  4. Entrar na pasta archive/ que foi criada.
  5. Dentro dela haverá a pasta chest_xray/.
  6. Copiar a pasta chest_xray e colar dentro de data/ do repositório:

         PS_Ligia_Desafio_Individual/
              └── data/
                    └── chest_xray/
                         ├── train/
                         └── test/

2. Dataset da Competição (submissão)

   1. Dataset da Competição (submissão)
      Lígia – Computer Vision: https://www.kaggle.com/competitions/ligia-compviz/data
   2. Clicar em Download All (lado direito inferior da tela): Será baixado o arquivo ligia-compviz.zip
   3. Descompactar o ligia-compviz.zip
   4. Descompactar o archive.zip
   5. Uma pasta chamada ligia-compviz será criada
   6. Copiar a pasta ligia-compviz e colar dentro de data/ do repositório:

          PS_Ligia_Desafio_Individual/
              └── data/
                    ├── chest_xray/
                    └── ligia-compviz/
                         ├── train.csv      
                         ├── test.csv
                              └── test_images/
                                   └── test_images/
   
Após esses passos, toda a estrutura de dados necessária estará pronta para uso local.

📓 Notebooks Incluídos

EDA.ipynb — análise exploratória dos dados e visualização das imagens.
Modelagem.ipynb — construção dos modelos, avaliação, validação e escolha do modelo final.
Inferencia.ipynb — inferência local, métricas finais e aplicação de interpretabilidade (Saliency e LIME).

⚠️ Observação: o notebook Modelagem.ipynb não roda localmente devido ao alto custo computacional.
Ele funciona como documentação completa do processo de treinamento e seleção do modelo.

🤖 Modelo Utilizado

O modelo final escolhido foi: EfficientNetB0 com Data Augmentation e Fine-Tuning Parcial
Backbone pré-treinado no ImageNet
Data augmentation leve (rotação, zoom, deslocamento, contraste)
Descongelamento parcial das camadas finais
Otimização fina com learning rate reduzido
O modelo final está salvo em: BestModel/best_model.keras

🔧 Como Executar o Projeto Localmente
1. Instalar Dependências
   Recomenda-se utilizar Python 3.10.
    ```bash
      pip install -r requirements.txt
    ```
   Este modelo é utilizado tanto no notebook de inferência quanto no script de geração de submissão.

2. Abrir o Projeto

  ```bash
      pip install -r requirements.txt
        
