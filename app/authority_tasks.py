# app/authority_tasks.py
from typing import Optional

# O dicionário gigante foi removido! 
# A inteligência agora é dinâmica baseada no tema e no formato.
AUTHORITY_TASKS = {}

def find_task_prompt(agent_key: str, title: str) -> Optional[str]:
    """
    Função depreciada. Retorna sempre None pois os prompts específicos 
    foram substituídos pela injeção dinâmica de Formato e Tema em ai.py.
    """
    return None