```
root/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
├── data/
│   ├── aerofoil_shapes/
│   └── results/
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── visualisations.ipynb
├── src/ ✅
│   ├── __init__.py
│   ├── aerofoil_env/ ✅
│   │   ├── __init__.py
│   │   ├── env.py
│   │   ├── xfoil_interface.py
│   │   ├── naca_airfoil.py ✅
│   │   └── utils.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── train_agent.py
│   │   └── evaluate_agent.py
│   └── models/
│       ├── __init__.py
│       └── saved_models/
├── scripts/
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── generate_plots.py
├── tests/
│   ├── __init__.py
│   ├── test_env.py
│   ├── test_xfoil.py
│   └── test_agents.py
└── docs/
    ├── index.md
    ├── setup_guide.md
    └── project_report.md
```