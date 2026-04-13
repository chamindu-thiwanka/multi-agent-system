from config import get_settings, validate_config

print('Testing config...')
settings = get_settings()
print(f'LLM Model: {settings.llm_model_name}')
print(f'Embedding Model: {settings.embedding_model_name}')
print(f'Chroma Host: {settings.chroma_host}:{settings.chroma_port}')
print()
validate_config()
