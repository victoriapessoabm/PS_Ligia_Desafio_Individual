🩺 Desafio Lígia – Classificação de Pneumonia em Raios-X

Este repositório contém todo o fluxo de desenvolvimento de uma solução de classificação binária de pneumonia em radiografias, incluindo análise exploratória, preparação dos dados, modelagem, interpretabilidade, inferência e geração da submissão final.
Todo o projeto foi configurado para rodar localmente, usando apenas caminhos relativos e mantendo portabilidade independente da máquina utilizada.

🧬 Clonar o Repositório

    git clone https://github.com/victoriapessoabm/PS_Ligia_Desafio_Individual.git
    cd PS_Ligia_Desafio_Individual

📥 Baixar e Organizar os Dados em data/
1. Dataset de Raios-X Rotulados (treino / validação / interpretação)

  1. Acessar o dataset no Kaggle:
      Labeled Chest X-Ray Images: https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
  2. Clicar em Download (será baixado um arquivo archive.zip).
  3. Descompactar o archive.zip.
  4. Entrar na pasta archive/ que foi criada.
  5. Dentro dela haverá a pasta chest_xray/.
  6. Copiar a pasta chest_xray e colar dentro de data/ do repositório:
  
  Resultado esperado:

         PS_Ligia_Desafio_Individual/
              └── data/
                    └── chest_xray/
                         ├── train/
                         └── test/

2. Baixar e Organizar os Dados em data/

   1. Dataset da Competição (submissão)
      Lígia – Computer Vision: https://www.kaggle.com/competitions/ligia-compviz/data
   2. Clicar em Download All (lado direito inferior da tela): Será baixado o arquivo ligia-compviz.zip
   3. Descompactar o ligia-compviz.zip
   4. Descompactar o archive.zip
   5. Uma pasta chamada ligia-compviz será criada
   6. Copiar a pasta ligia-compviz e colar dentro de data/ do repositório:

   Resultado esperado: 

          PS_Ligia_Desafio_Individual/
              └── data/
                    ├── chest_xray/
                    └── ligia-compviz/
                         ├── train.csv      
                         ├── test.csv
                              └── test_images/
                                   └── test_images/
   
Após esses passos, toda a estrutura de dados necessária estará pronta para uso local.

📁 Estrutura esperada para o Repositório

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

📓 Notebooks Incluídos

- EDA.ipynb — análise exploratória dos dados e visualização das imagens.
- Modelagem.ipynb — construção dos modelos, avaliação, validação e escolha do modelo final.
- Inferencia.ipynb — inferência local, métricas finais e aplicação de interpretabilidade (Saliency e LIME).

⚠️ Observação: o notebook Modelagem.ipynb não roda localmente devido ao alto custo computacional.
Ele funciona como documentação completa do processo de treinamento e seleção do modelo.

🤖 Modelo Utilizado

- O modelo final escolhido foi: EfficientNetB0 com Data Augmentation e Fine-Tuning Parcial
- Backbone pré-treinado no ImageNet;
- Data augmentation leve (rotação, zoom, deslocamento, contraste);
- Descongelamento parcial das camadas finais;
- Otimização fina com learning rate reduzido;
- O modelo final está salvo em: BestModel/best_model.keras

Este modelo é utilizado tanto no notebook de inferência quanto no script de geração de submissão.

🔧 Como Executar o Projeto Localmente

1. Instalar Dependências
   Recomenda-se utilizar Python 3.10.
   Na raiz do projeto: 
    ```bash
      pip install -r requirements.txt
    ```
   Se houver mais de uma versão de Python instalada, usar explicitamente:
   ```bash
      python3.10 -m pip install -r requirements.txt
    ```
   
2. Abrir o Projeto

    ```bash
      cd PS_Ligia_Desafio_Individual
      code .
    ```
3. Executar os notebooks localmente

   - Abrir os notebooks em Notebooks/: EDA.ipynb e Inferencia.ipynb
   - Selecionar o kernel Python 3.10 (ou o ambiente onde as dependências foram instaladas).
   - Executar as células em sequência.

📦 Geração da Submissão para o Kaggle
   - Para executar:

            python3 src/generateSubmission.py
    
   - Esse script:
        - Localiza automaticamente a raiz do projeto;
        - Carrega BestModel/best_model.keras;
        - Lê data/ligia-compviz/test.csv;
        - Monta os caminhos das imagens em data/ligia-compviz/test_images/test_images/;
        - Executa a inferência sobre todas as imagens de teste;
        - Salva o arquivo final em: 

                 Submission/submission_membros.csv
                 

🔁 Portabilidade e Observações

- Não há caminhos absolutos no código;
- A raiz do projeto é identificada dinamicamente dentro dos notebooks e scripts;
- Todos os acessos a arquivos utilizam caminhos relativos à pasta do repositório;
- Mantendo a estrutura de diretórios e instalando as dependências, o projeto pode ser executado em qualquer ambiente compatível com Python 3.10.

- Este repositório documenta o ciclo completo da solução: EDA → preparação dos dados → modelagem → interpretabilidade → inferência → submissão.
