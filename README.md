# Analise e Previsao de Sucesso em Dietas

Este projeto combina dados de dietas e pacientes para executar analises basicas, agrupamento e modelos de previsao para alteracao de peso.

## Estrutura do Projeto

```text
.
├── data/                  # Arquivos CSV brutos e processados
├── src/                   # Codigo de processamento, analise e modelagem
├── main.py                # Arquivo principal que executa todo o pipeline
├── requirements.txt       # Bibliotecas Python necessarias
├── eda_results/           # Resultados da analise exploratoria de dados
├── clustering_results/    # Resultados do agrupamento
└── modeling_results/      # Resultados da modelagem
```

## Instalacao

Instale os pacotes necessarios:

```bash
pip install -r requirements.txt
```

## Uso

Execute todo o pipeline do projeto:

```bash
python main.py
```

Este comando ira:

1. Unir os arquivos CSV.
2. Limpar e normalizar os dados.
3. Criar graficos de analise exploratoria de dados.
4. Testar padroes especificos nos dados.
5. Executar agrupamento com K-Means.
6. Treinar modelos para prever alteracao de peso.

## Resultados

Apos executar o projeto, os arquivos de saida sao criados em:

- `data/merged_data.csv`
- `data/cleaned_data.csv`
- `eda_results/`
- `clustering_results/`
- `modeling_results/`

## Bibliotecas Utilizadas

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

# English

## Diet Success Analysis & Prediction

This project combines diet and patient data to run basic analysis, clustering, and prediction models for weight change.

## Project Structure

```text
.
├── data/                  # Raw and processed CSV files
├── src/                   # Data processing, analysis, and modeling code
├── main.py                # Main file that runs the full pipeline
├── requirements.txt       # Required Python libraries
├── eda_results/           # Exploratory data analysis outputs
├── clustering_results/    # Clustering outputs
└── modeling_results/      # Modeling outputs
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the full project pipeline:

```bash
python main.py
```

This command will:

1. Merge the CSV files.
2. Clean and normalize the data.
3. Create exploratory data analysis plots.
4. Test specific data patterns.
5. Run K-Means clustering.
6. Train models to predict weight change.

## Outputs

After running the project, output files are created in:

- `data/merged_data.csv`
- `data/cleaned_data.csv`
- `eda_results/`
- `clustering_results/`
- `modeling_results/`

## Libraries Used

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
