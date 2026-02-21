🩺 Desafio Lígia – Classificação de Pneumonia em Raios-X

Este repositório contém todo o fluxo de desenvolvimento de uma solução de classificação binária de pneumonia em radiografias, incluindo análise exploratória, preparação dos dados, modelagem, interpretabilidade, inferência e geração da submissão final.
Todo o projeto foi configurado para rodar localmente, usando apenas caminhos relativos e mantendo portabilidade independente da máquina utilizada.

🧬 Clonar o Repositório

    git clone https://github.com/victoriapessoabm/PS_Ligia_Desafio_Individual.git
    cd PS_Ligia_Desafio_Individual

📥 Baixar e Organizar os Dados em data/

1. Dataset de Raios-X Rotulados (treino / validação / interpretação/ inferência)
- Acessar o dataset no Kaggle:
      Labeled Chest X-Ray Images:
      https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images
- Clicar em Download no canto superior direito (será baixado um arquivo archive.zip)
- Descompactar o archive.zip
- Entrar na pasta archive/ que foi criada
- Dentro dela haverá a pasta chest_xray/
- Copiar a pasta chest_xray e colar dentro de data/ do repositório:
  
  Resultado esperado:

         PS_Ligia_Desafio_Individual/
              └── data/
                    └── chest_xray/
                         ├── train/
                         └── test/
                    └── dataset.csv

2. Dataset da competição não-rotulado (usado para criar submission_membros.csv) 
- Dataset da Competição (submissão)
      Lígia – Computer Vision:
      https://www.kaggle.com/competitions/ligia-compviz
- Clicar em Download All (lado direito na parte inferior da tela): Será baixado o arquivo ligia-compviz.zip
- Descompactar o ligia-compviz.zip
- Descompactar o archive.zip
- Uma pasta chamada ligia-compviz será criada
- Copiar a pasta ligia-compviz e colar dentro de data/ do repositório:

   Resultado esperado: 

          PS_Ligia_Desafio_Individual/
              └── data/
                    ├── chest_xray/
                    └── dataset.csv
                    └── ligia-compviz/
                         ├── train.csv
                         ├── test.csv   
                         ├── train      
                         ├── test
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

🔧 Como Executar o Projeto Localmente

1. Instalar Dependências
   Recomenda-se utilizar Python 3.10.
   Na raiz do projeto: 
    ```bash
      pip install -r requirements.txt
    ```
   OBS: se houver mais de uma versão de Python instalada, usar explicitamente:
   ```bash
      python3.10 -m pip install -r requirements.txt
    ```
   
2. Abrir o Projeto

    ```bash
      cd PS_Ligia_Desafio_Individual
      code .
    ```
3. Executar os notebooks localmente

   - Abrir os notebooks em Notebooks/EDA.ipynb e Notebooks/Inferencia.ipynb
   - Clicar na primeira célula do primeiro notebook escolhido para rodar
   - Selecionar o Python Enviroments e, sem seguida, selecionar o kernel Python 3.10 (ou o ambiente onde as dependências foram instaladas)
   - Por fim, executar todas as outras células em sequência (Run All Cells)

📦 Geração da Submissão para o Kaggle
   - Para executar, você deve executar o comando a partir da raiz do projeto, que é o diretório: PS_Ligia_Desafio_Individual/
   - Na raiz, execute: 
   
            python3 src/generateSubmission.py
    
   - Esse script:
        - Localiza automaticamente a raiz do projeto;
        - Carrega BestModel/best_model.keras
        - Lê data/ligia-compviz/test.csv
        - Monta os caminhos das imagens em data/ligia-compviz/test_images/test_images/
        - Executa a inferência sobre todas as imagens de teste;
        - Salva o arquivo final em: 

                 Submission/submission_membros.csv
                 

🔁 Portabilidade e Observações

- Não há caminhos absolutos no código;
- A raiz do projeto é identificada dinamicamente dentro dos notebooks e scripts;
- Todos os acessos a arquivos utilizam caminhos relativos à pasta do repositório;
- Mantendo a estrutura de diretórios e instalando as dependências, o projeto pode ser executado em qualquer ambiente compatível com Python 3.10.

📄 Função de cada arquivo do repositório

    BestModel/best_model.keras -> modelo final treinado (EfficientNetB0 com fine-tuning parcial), utilizado para todas as inferências e submissões.
    data/
        chest_xray/ — imagens rotuladas usadas para EDA e notebook de inferência
        ligia-compviz/ — arquivos oficiais da competição (são utilizados para gerar arquivo de submissão: test.csv + imagens de teste)
        dataset.csv – dataset tratado consolidando caminhos das imagens, rótulos e splits
    ImagePreprocessing/
        imagePipeline.py – métodos para o tratamento das imagens para prepará-las para os modelos
        modelBuilder.py – uso dos métodos de imagePipeline.py para preparar dados para os modelos utilizados: baseline - CNN simples, EfficientNet;
        preprocessing.py – preparação do dataset (limpeza, splits, ajuste de caminhos)
        generate_csv.py – gera o dataset.csv a partir das pastas de imagens.
    Interpretability/
        saliency.py – Saliency Maps e visualização
        lime.py – LIME para imagens e visualização
    Notebooks/
        EDA.ipynb – análise exploratória
        Modelagem.ipynb – treinamento e avaliação dos modelos
        Inferencia.ipynb – inferência local e interpretabilidade
    Submission/
        submission.csv - meu resultado de submissão para o Kaggle 
        submission_membros.csv - resultado gerado ao rodar script generateSubmission.py
    src/
        generateSubmission.py - Gera automaticamente Submission/submission_membros.csv usando o modelo final
    requirements.txt - lista de dependências do projeto

📓 Notebooks 

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

🧾 Conclusão

- Três modelos foram testados ao longo do projeto: 
    - CNN simples (baseline) — serviu como ponto de partida, oferecendo uma referência inicial de desempenho.
    - EfficientNetB0 pré-treinado, sem augmentation e sem fine-tuning — apresentou melhora imediata em relação ao baseline devido ao uso de pesos pré-treinados.
    - EfficientNetB0 com Data Augmentation e Fine-Tuning Parcial (modelo final) — alcançou o melhor desempenho geral, combinando robustez, estabilidade e boa capacidade de generalização.
- O modelo final foi salvo em BestModel/best_model.keras e é utilizado tanto no notebook de inferência quanto no script de geração da submissão (generateSubmission.py)
- A estrutura do projeto utiliza exclusivamente caminhos relativos, garantindo que todo o fluxo possa ser executado em qualquer máquina, desde que a organização das pastas seja mantida e as dependências sejam instaladas.
- O notebook de Modelagem documenta todo o processo de treinamento, enquanto os notebooks de EDA e Inferência podem ser executados localmente. No Inferencia.ipynb estão incluídas também as etapas de interpretabilidade com Saliency e LIME, permitindo visualizar e analisar como o modelo final toma suas decisões.

✨ Autoria
Projeto desenvolvido por Victória Pessoa Barbosa Matos na segunda etapa do processo seletivo da Ligia (Liga de Inteligência Artificial do CIn - UFPE) 
Desafio Lígia – Visão Computacional
