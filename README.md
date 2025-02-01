# porcelain-mark-recognizer

Сервис для определения марки фарфорового изделия

### Запуск без docker

> Все команды исполняются из корня проекта

0. **Подготовка окружения**

```bash
pip install uv
uv venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

1. **Запуск backend**

```bash
python3 -m search-engine.main
```

2. **Запуск streamlit (в другом терминале)**

```bash
. .venv/bin/activate
streamlit run frontend.streamlit_app --server.port=8501 --server.address=0.0.0.0
```

### Запуск с помощью docker

0. Убедиться, что docker установлен
1. В корне проекта

```bash
sudo docker compose up
```

