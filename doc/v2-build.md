src/
├── client/
│   ├── __init__.py           # Export du client principal
│   ├── client.py             # Classe principale LLMClient
│   ├── config.py             # Configuration (base URL, timeout, etc.)
│   │
│   ├── endpoints/
│   │   ├── __init__.py
│   │   ├── base.py           # Classe de base pour les endpoints
│   │   ├── root.py           # GET /
│   │   ├── models.py         # GET /models, POST /models/load, POST /models/unload
│   │   ├── health.py         # GET /health, GET /health/<gpu_id>
│   │   ├── chat.py           # POST /chat/<gpu_id>
│   │   ├── completion.py     # POST /completion/<gpu_id>
│   │   └── logs.py           # GET /logs/stream, /logs/history, /logs/stats
│   │
│   ├── exceptions.py         # Exceptions personnalisées
│   └── types.py              # Types/classes de données (ModelInfo, GPUStatus, etc.)
│
└── main.py                   # Votre API Flask