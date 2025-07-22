# AI Assistant for Risk Assessment and Mitigation in Banking

![Python 3.12.x](https://img.shields.io/badge/Python-3.12.x-blue) 
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green) 
![LangGraph](https://img.shields.io/badge/LangGraph-Enabled-orange) 
![AI Models](https://img.shields.io/badge/AI%20Models-Gemini%20%7C%20Llama%203-9cf) 
![Baseline Models](https://img.shields.io/badge/Baselines-XGBoost%20%7C%20LR%20%7C%20K--means-lightgrey) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced AI Assistant developed with **LangChain**, **LangGraph**, and **Python 3.12** to revolutionize risk assessment and mitigation in the banking sector. This project leverages Generative AI to provide real-time, intelligent support for bankers.

-----

## 📖 About The Project

Traditional risk assessment and mitigation in banking are often slow, labor-intensive, and costly. This project addresses these challenges by creating a powerful **AI Assistant** that acts as a specialized financial expert.

By fine-tuning large language models (LLMs) on specialized financial datasets, the assistant provides bankers with highly accurate, consistent, and on-demand support for critical risk management tasks. The goal is to enhance decision-making, improve efficiency, and strengthen the bank's risk management framework.

### Key Features 🎯

  * **Real-time Fraud Detection:** Identify and flag fraudulent transactions as they happen.
  * **Credit Risk Early Warning:** Proactively detect early signs of credit default risk.
  * **Risk Mitigation Suggestions:** Provide actionable recommendations to mitigate identified risks.
  * **Human-like Interaction:** A user-friendly chat interface for seamless interaction.

-----

## 🛠️ Technology Stack

This project is built using a modern, powerful stack designed for building state-of-the-art AI applications:

  * **Backend:** Python 3.12
  * **AI Orchestration:** LangChain & LangGraph
  * **Generative AI Models:**
      * **Commercial:** Gemini 2.5
      * **Open Source:** Llama 3 (or Qwen3)
  * **Baseline ML Models:** Logistic Regression, XGBoost, K-means
  * **Frontend:** Prototyped with a simple ChatGPT-like interface.

-----

## 🔬 Methodology & Objectives

The research aims to design, build, and benchmark a Generative AI assistant against traditional machine learning approaches.

### Datasets

Both the GenAI and traditional models are trained and evaluated on the same standardized datasets:

1.  [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
2.  [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
3.  [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

### Research Objectives

1.  **Design & Implement AI Assistant:** Design a robust system architecture and implement an AI Agent using fine-tuned **Gemini 2.5** and **Llama 3** models.
2.  **Establish Baselines:** Train traditional ML models (**Logistic Regression, XGBoost, K-means**) on the same datasets to create performance baselines for speed, latency, and accuracy.
3.  **Develop Prototype UI:** Build a simple, interactive chat interface to allow bankers to communicate with the AI assistant.

-----

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

  * Python 3.12 or later
  * Pip package manager
  * An API key for the Gemini model

### Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py         # Pydantic settings for environment variables
│   ├── graph.py          # Core LangGraph agent definition
│   ├── llms.py           # LLM initializations
│   ├── main.py           # FastAPI app and Gradio UI entrypoint
│   ├── state.py          # LangGraph state definition
│   └── tools/
│       ├── __init__.py
│       ├── databricks_rag.py # RAG tool for Databricks
│       └── ml_models.py      # Tools for traditional ML models
├── data_processing/
│   ├── __init__.py
│   ├── train_credit_model.py # Script to train a dummy credit risk model
│   └── train_fraud_model.py  # Script to train a dummy fraud detection model
├── models/
│   └──.gitkeep          # Directory for saved ML models
├── tests/
│   └──.gitkeep          # Directory for tests
├──.env.example          # Example environment variables
├──.gitignore
├── pyproject.toml        # Poetry project configuration
└── poetry.lock
```

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install Poetry:**
    If you don't have Poetry, follow the instructions at [python-poetry.org](https://python-poetry.org/docs/#installation).

3.  **Install dependencies:**
    ```bash
    poetry install
    ```

4.  **Set up environment variables:**
    Copy the example `.env` file and fill in your API keys and configuration details.
    ```bash
    cp.env.example.env
    # Now edit the.env file with your credentials
    ```

5.  **Train dummy ML models (Optional):**
    To create the placeholder model files that the tools will load, run the training scripts.
    ```bash
    poetry run python data_processing/train_fraud_model.py
    poetry run python data_processing/train_credit_model.py
    ```

### Running the Application

1.  **Start the FastAPI server:**
    ```bash
    poetry run start
    ```

2.  **Access the application:**
    *   The Gradio UI will be available at `http://127.0.0.1:8000`.
    *   The API documentation (Swagger UI) is at `http://127.0.0.1:8000/docs`.
-----

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

-----

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
