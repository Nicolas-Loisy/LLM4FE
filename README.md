# LLM4FE Project

This project aims to develop an automated pipeline for Feature Engineering (FE), AutoML, and Benchmarking, orchestrated by a central orchestrator. It dynamically transforms a dataset using a LLM, trains an AutoML model, and benchmarks its performance, storing results in a versioned structure.

## Project Structure

```
/LLM4FE
│
├── 📂 data/  # Contains datasets and saved models
│   ├── models/  # Trained models
│   ├── logs/  # Execution logs
│
├── 📂 src/
│   ├── __init__.py
│
│   ├── 📂 orchestrator/  # Central pipeline management module
│   │   ├── __init__.py
│   │   ├── orchestrator.py  # Pipeline coordination
│   │   ├── config.py  # Configuration management
│
│   ├── 📂 feature_engineering/  # Data transformation module
│   │   ├── __init__.py
│   │   ├── fe_pipeline.py  # Transformation execution
│   │   ├── fe_factory.py  # Manage transformations dynamically
│   │
│   │   ├── 📂 transformations/  # Specific transformations
│   │   │   ├── __init__.py
│   │   │   ├── base_transform.py  # Parent class for all transformations
│   │   │   ├── scaling.py  # Scaling transformations
│   │   │   ├── encoding.py  # Categorical variable encoding
│   │   │   ├── text_processing.py  # Text processing (TF-IDF, embeddings, etc.)
│   │   │   ├── math_operations.py  # Math functions (log, mean, etc.)
│
│   ├── 📂 llm/  # LLM call management
│   │   ├── __init__.py
│   │   ├── llm_factory.py  # Support for multiple LLM models
│   │   ├── openwebui_client.py  # OpenWebUI API client
│
│   ├── 📂 automl/  # Model training
│   │   ├── __init__.py
│   │   ├── automl_pipeline.py  # AutoML process execution
│
│   ├── 📂 benchmark/  # Model evaluation
│   │   ├── __init__.py
│   │   ├── benchmark.py  # Model score calculation
│
├── requirements.txt
├── README.md
├── .gitignore
└── run_pipeline.py  # Main script launching the pipeline
```

## Technologies Used

- **Python**
- **OpenWebUI API** for LLM calls
- **Pydantic** for validating transformation JSONs
- **Scikit-learn, Auto-sklearn, TPOT** for AutoML
- **Pandas, NumPy** for data manipulation
