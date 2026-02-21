'''Конфигурация проекта'''

from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    '''Основные настройки проекта'''
    project_root: Path = Path(__file__).parent.parent
    results_dir: Path = project_root / "results"
    raw_results_dir: Path = results_dir / "raw"
    tables_dir: Path = results_dir / "tables"
    plots_dir: Path = results_dir / "plots"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"
    ollama_temperature: float = 0.0
    ollama_timeout: int = 120

    mcp_server_name: str = "nlp-tools"
    mcp_server_host: str = "localhost"
    mcp_server_port: int = 8765

    max_samples: int = 200
    random_seed: int = 42

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "NLP_RESEARCH_"
    
    def ensure_dirs(self) -> None:
        """Создать директории для результатов, если не существуют"""
        for directory in (
            self.raw_results_dir,
            self.tables_dir,
            self.plots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()