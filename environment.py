from pathlib import Path 
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Путь до весов YOLO 
    yolo_model_path: str
    yolo_conf_threshold: float

    # Настройки поискового движка
    search_model_path: str
    search_images_dir: str
    search_top_k: int
    search_similarity_threshold: float

    # Настройки API
    api_host: str
    api_port: str
    
    # Настройки Frontend
    front_host: str
    front_port: str
    
    class Config:
        env_file = ".env"
        

settings = Settings()