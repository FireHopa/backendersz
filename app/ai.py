from __future__ import annotations

import json
import httpx
import io
import uuid
from typing import List, Dict, Any

from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TRANSCRIBE_MODEL, SERPER_API_KEY, SERPER_GL, SERPER_HL, SERPER_LOCATION, ENABLE_WEB_SEARCH, WEB_SEARCH_MAX_RESULTS
from .prompts import BUILDER_SYSTEM, GLOBAL_AIO_AEO_GEO, COMPETITOR_FINDER_SYSTEM, COMPETITION_ANALYSIS_SYSTEM, AUTHORITY_ASSISTANT_SYSTEM

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def _require_key():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY não configurada. Edite o seu .env com uma chave válida.")

def build_robot_from_briefing(briefing: Dict[str, Any]) -> Dict[str, str]:
    _require_key()

    prompt_user = {
        "briefing": briefing,
        "output_rules": {
            "language": "pt-BR",
            "must_include": ["AIO", "AEO", "GEO"],
            "must_be_json": True,
        },
    }

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": BUILDER_SYSTEM + "\n\n" + GLOBAL_AIO_AEO_GEO},
            {"role": "user", "content": json.dumps(prompt_user, ensure_ascii=False)}
        ],
        temperature=0.4,
        max_tokens=900,
        response_format={"type": "json_object"} # Garante que a OpenAI devolva JSON puro
    )

    text = (resp.choices[0].message.content or "").strip()
    
    # tolerate fenced json caso o modelo ainda teime em mandar markdown
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    data = json.loads(text)
    title = str(data.get("title", "")).strip() or f"Robô {briefing.get('nicho','')}".strip() or "Robô"
    system_instructions = str(data.get("system_instructions", "")).strip()
    
    if not system_instructions:
        raise RuntimeError("O construtor não retornou system_instructions.")

    return {"title": title, "system_instructions": system_instructions}


def _format_web_context(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["FONTES DA WEB (para fatos; cite por [n]):"]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snip = (r.get("snippet") or "").strip()
        lines.append(f"[{i}] {title} — {url}")
        if snip:
            lines.append(f"    {snip}")
    return "\n".join(lines).strip()


def _filter_domains(results: list[dict], allowed_domains: list[str] | None) -> list[dict]:
    if not allowed_domains:
        return results
    allowed = {d.lower().lstrip("www.") for d in allowed_domains if d}
    if not allowed:
        return results
    from urllib.parse import urlparse
    out = []
    for r in results:
        url = str(r.get("url") or "")
        try:
            host = urlparse(url).netloc.lower().lstrip("www.")
        except Exception:
            host = ""
        if host and any(host == d or host.endswith("."+d) for d in allowed):
            out.append(r)
    return out


def chat_with_robot(
    system_instructions: str,
    history: List[Dict[str, str]],
    user_message: str,
    *,
    use_web: bool = False,
    web_max_results: int | None = None,
    web_allowed_domains: list[str] | None = None,
) -> str:
    _require_key()

    max_results = int(web_max_results or WEB_SEARCH_MAX_RESULTS or 5)
    max_results = max(1, min(max_results, 20))

    web_block = ""
    if (use_web or ENABLE_WEB_SEARCH) and SERPER_API_KEY:
        results = _serper_search(user_message, max_results=max_results)
        results = _filter_domains(results, web_allowed_domains)[:max_results]
        web_block = _format_web_context(results)

    input_msgs = history[-20:]
    if web_block:
        input_msgs = input_msgs + [{"role": "user", "content": web_block}]
    input_msgs = input_msgs + [{"role": "user", "content": user_message}]

    sys_msg = system_instructions + "\n\nSe usar FONTES DA WEB, cite no texto com [n] (ex: [1]) e não invente links."

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": sys_msg}] + input_msgs,
        temperature=0.6,
        max_tokens=1200,
    )
    return (resp.choices[0].message.content or "").strip()


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcreve áudio para texto (pt-BR)."""
    _require_key()
    if not audio_bytes:
        raise RuntimeError("Áudio vazio.")
    
    try:
        resp = client.audio.transcriptions.create(
            model=OPENAI_TRANSCRIBE_MODEL,
            file=(filename, audio_bytes, "audio/webm"),
            language="pt",
        )
    except Exception as e:
        # fallback clássico
        if OPENAI_TRANSCRIBE_MODEL != "whisper-1":
            try:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(filename, audio_bytes, "audio/webm"),
                    language="pt",
                )
            except Exception:
                raise
        else:
            raise
    
    text = getattr(resp, "text", None)
    if text is None and isinstance(resp, dict):
        text = resp.get("text")
    return (text or "").strip()


def _serper_search(query: str, max_results: int = 10) -> list[dict]:
    if not SERPER_API_KEY:
        return []
    url = "[https://google.serper.dev/search](https://google.serper.dev/search)"
    payload = {
        "q": query,
        "gl": SERPER_GL or "br",
        "hl": SERPER_HL or "pt-br",
        "location": SERPER_LOCATION or "Brazil",
        "num": max(1, min(int(max_results), 20)),
    }
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=20.0) as client_http:
            r = client_http.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []
    organic = data.get("organic") or []
    out: list[dict] = []
    for it in organic[:max_results]:
        link = str(it.get("link") or "").strip()
        if not link:
            continue
        out.append({
            "title": str(it.get("title") or link).strip(),
            "url": link,
            "snippet": str(it.get("snippet") or "").strip(),
        })
    return out


def find_competitors(profile: dict) -> dict:
    niche = str(profile.get("niche") or profile.get("segmento") or "").strip()
    region = str(profile.get("region") or profile.get("cidade_estado") or "").strip()
    services = str(profile.get("services") or profile.get("servicos") or profile.get("offer") or "").strip()
    audience = str(profile.get("audience") or profile.get("publico_alvo") or "").strip()

    missing = [k for k,v in {
        "segmento/nicho": niche,
        "cidade/estado": region,
        "serviços": services,
        "público-alvo": audience,
    }.items() if not v]
    if missing:
        return {"suggestions": [], "sources": [], "note": f"Informações insuficientes para buscar concorrentes. Falta: {', '.join(missing)}", "data_quality":"incomplete"}

    queries = [
        f"{niche} {services} {region}",
        f"{niche} {region} site",
        f"{niche} {region} instagram",
    ]
    sources: list[dict] = []
    seen = set()
    for q in queries:
        for s in _serper_search(q, max_results=10):
            url = s.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append(s)
        if len(sources) >= 12:
            break

    if not sources:
        return {
            "suggestions": [],
            "sources": [],
            "note": "Sem resultados de busca web. Verifique SERPER_API_KEY ou refine cidade/segmento/serviços.",
            "data_quality": "incomplete",
        }

    prompt_input = {
        "profile": {"niche": niche, "region": region, "services": services, "audience": audience},
        "sources": sources[:12],
        "rules": {"max_competitors": 3, "local_first": True},
    }

    try:
        if OPENAI_API_KEY:
            from .openai_client import chat_json
            data = chat_json(
                system=COMPETITOR_FINDER_SYSTEM,
                user=json.dumps(prompt_input, ensure_ascii=False),
                model=OPENAI_MODEL,
            )
        else:
            data = {}
    except Exception:
        data = {}

    suggestions = data.get("suggestions") if isinstance(data, dict) else None
    if not isinstance(suggestions, list):
        suggestions = []
        for s in sources:
            suggestions.append({
                "name": s.get("title") or s.get("url"),
                "website_url": s.get("url"),
                "instagram": None,
                "reason": "Encontrado em busca pública",
                "confidence": 0.55,
            })
            if len(suggestions) >= 3:
                break

    return {"suggestions": suggestions[:3], "sources": sources[:8], "note": "Sugestões geradas via busca (Serper).", "data_quality":"ok"}


def build_competition_result(company: dict, competitors: list[dict]) -> dict:
    payload = {"company": company, "competitors": competitors}
    try:
        if OPENAI_API_KEY:
            from .openai_client import chat_json
            res = chat_json(
                system=COMPETITION_ANALYSIS_SYSTEM,
                user=json.dumps(payload, ensure_ascii=False),
                model=OPENAI_MODEL,
            )
            if isinstance(res, dict):
                return res
    except Exception:
        pass

    def mk_signals():
        return {
            "presence": 50,
            "offer_clarity": 50,
            "communication": 50,
            "content_frequency": 40,
            "positioning": 50,
            "perceived_authority": 45,
        }

    company_out = {
        "name": company.get("nome_empresa") or company.get("company_name") or "Sua empresa",
        "niche": company.get("segmento") or company.get("niche"),
        "region": company.get("cidade_estado") or company.get("region"),
        "services": company.get("servicos") or company.get("services"),
        "audience": company.get("publico_alvo") or company.get("audience"),
        "signals": mk_signals(),
        "notes": ["Relatório gerado em modo básico (sem IA)."],
    }
    comps_out = []
    for c in competitors[:3]:
        comps_out.append({
            "name": c.get("name") or "Concorrente",
            "website_url": c.get("website_url"),
            "instagram": c.get("instagram"),
            "signals": mk_signals(),
            "highlights": ["Presença encontrada em busca pública."],
            "gaps": ["Dados detalhados não disponíveis."],
        })

    return {
        "company": company_out,
        "competitors": comps_out,
        "comparisons": {"bar": [], "radar": []},
        "insights": [],
        "recommendations": [],
        "transparency": {"limitations": ["Modo básico sem IA ativa."]},
    }


def authority_assistant(
    *,
    robot_system_instructions: str,
    user_message: str,
    history: List[Dict[str, str]] | None = None,
    authority_edits_history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    _require_key()
    history = history or []
    
    clean_hist = []
    for m in history[-20:]:
        role = str(m.get("role", "")).strip().lower()
        if role not in ("user", "assistant"):
            continue
        clean_hist.append({"role": role, "content": str(m.get("content",""))})

    payload = {
        "system_instructions_current": robot_system_instructions,
        "assistant_history": clean_hist,
        "authority_edits_history": authority_edits_history or [],
        "user_message": user_message,
    }

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": AUTHORITY_ASSISTANT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        response_format={"type": "json_object"}
    )
    
    text = (resp.choices[0].message.content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    data = json.loads(text)

    apply_change = bool(data.get("apply_change", False))
    updated = data.get("updated_system_instructions", None)
    if not apply_change:
        updated = None

    data["apply_change"] = apply_change
    data["updated_system_instructions"] = updated
    return data


AUTHORITY_AGENTS = {
    "site": {
        "name": "Caio Web",
        "type": "Agente Site",
        "instructions": "🎯 OBJETIVO DO AGENTE: Você é um Agente de Criação de Conteúdo para Sites. Seu papel é gerar conteúdos focados em Autoridade digital, clareza para o cliente e otimização para IA (AIO), Mecanismos de Resposta (AEO) e Buscas Locais (GEO).\n\n🧠 REGRAS DE QUALIDADE:\n- Escrever em português claro e profissional.\n- Evitar jargões vazios e promessas exageradas.\n- Focar em clareza, objetividade e leitura escaneável.\n- Não inventar anos, prémios ou números que não foram fornecidos.",
    },
    "google_business_profile": {
        "name": "Gabi Maps",
        "type": "Agente Perfil de Empresa no Google",
        "instructions": "🎯 OBJETIVO DO AGENTE: Você é um Agente de Autoridade Local e Semântica. Seu papel é transformar a presença da empresa num nó de autoridade local (GEO) e numa fonte de respostas precisas (AEO/AIO).\n\n🧠 REGRAS DE QUALIDADE:\n- Nada genérico ou promessas vazias.\n- Deixar sempre claro: O que faz, Para quem, Onde e Como resolve.\n- Não inventar endereços ou bairros se não forem fornecidos.",
    },
    "social_proof": {
        "name": "Rafa Reputação",
        "type": "Agente Prova social e reputação",
        "instructions": "🎯 OBJETIVO DO AGENTE: Você é um Agente de Autoridade, Confiança e Reputação. Seu papel é transformar casos de sucesso e contextos da empresa em ativos de conversão persuasivos e credíveis.\n\n🧠 REGRAS DE QUALIDADE:\n- Não inventar depoimentos fictícios com nomes de pessoas.\n- Focar no contexto da transformação (Situação antes -> O que foi feito -> Resultado).\n- Manter um tom humano e empático.",
    },
    "decision_content": {
        "name": "Duda Decisão",
        "type": "Agente Conteúdos de decisão",
        "instructions": "🎯 OBJETIVO DO AGENTE: Você é um Agente de Arquitetura de Decisão e Conversão. Seu papel é criar argumentos e respostas que reduzam a incerteza do cliente no fundo de funil, facilitando a decisão de compra.\n\n🧠 REGRAS DE QUALIDADE:\n- Focar em quebrar objeções de forma honesta e profissional.\n- Não atacar concorrentes diretamente por nome.\n- Ser direto nas respostas: a primeira frase já deve responder à dúvida principal.",
    },
    "instagram": {
        "name": "Bia Insta",
        "type": "Agente Instagram",
        "instructions": "🎯 OBJETIVO DO AGENTE: Você é um Agente de Posicionamento e Descoberta no Instagram. Seu papel é criar conteúdos que tragam clareza imediata e retenham a atenção do público da empresa.\n\n🧠 REGRAS DE QUALIDADE:\n- Textos prontos para uso em redes sociais (dinâmicos e envolventes).\n- Usar ganchos fortes nos primeiros 3 segundos (ou primeiras linhas).\n- Aplicar chamadas para ação (CTAs) simples e não forçadas.",
    },
    "linkedin": {
        "name": "Leo B2B",
        "type": "Agente LinkedIn",
        "instructions": "🎯 OBJETIVO DO AGENTE: Você é um Agente de Posicionamento Profissional e Autoridade no LinkedIn (B2B). Seu papel é consolidar a entidade da empresa ou especialista como referência no seu nicho.\n\n🧠 REGRAS DE QUALIDADE:\n- Tom executivo, profissional e direto.\n- Evitar jargões motivacionais genéricos.\n- Foco em utilidade técnica, partilha de processos e visão de mercado.",
    },
    "youtube": {
        "name": "Yuri Vídeos",
        "type": "Agente YouTube",
        "instructions": "🎯 OBJETIVO DO AGENTE: Você é um Agente de Autoridade em Vídeo e Descoberta no YouTube. Seu papel é roteirizar conteúdos focados em intenção de busca (SEO/AEO), ensinando e convertendo através de vídeos.\n\n🧠 REGRAS DE QUALIDADE:\n- Evitar \"clickbait\" irreal.\n- Estruturar roteiros com: Gancho forte -> Resposta direta -> Explicação/Desenvolvimento -> CTA.\n- Linguagem falada (mais natural do que um texto escrito).",
    },
    "tiktok": {
        "name": "Tati Trend",
        "type": "Agente TikTok",
        "instructions": "🎯 OBJETIVO DO AGENTE: Você é um Agente de Descoberta Rápida em Vídeo Curto. Seu papel é criar guiões super rápidos e diretos que retenham a atenção da primeira à última palavra.\n\n🧠 REGRAS DE QUALIDADE:\n- Guiões ágeis e hiper fofados.\n- Corte o excesso de introduções (\"Olá, o meu nome é...\").\n- Entregue valor logo no primeiro segundo do texto.",
    },
    "cross_platform_consistency": {
        "name": "Cris Consistência",
        "type": "Agente Consistência entre plataformas",
        "instructions": "🎯 OBJETIVO DO AGENTE: Você é um Agente de Governança de Identidade e Consistência Semântica. Seu papel é auditar e unificar a forma como a empresa se apresenta para que seja compreendida como uma única \"Entidade\" forte por humanos e IAs.\n\n🧠 REGRAS DE QUALIDADE:\n- Foco extremo na padronização de nomenclatura, serviços e propostas de valor.\n- Clareza e objetividade cirúrgicas.",
    },
    "external_mentions": {
        "name": "Nina Menções",
        "type": "Agente Menções externas",
        "instructions": "🎯 OBJETIVO DO AGENTE: Você é um Agente de Preparação para Menções e Citações Externas (PR e Link Building). Seu papel é criar ativos para quando a empresa for citada por portais, parceiros e imprensa.\n\n🧠 REGRAS DE QUALIDADE:\n- Tom estritamente jornalístico e institucional.\n- Sem exageros comerciais ou linguagem de vendas direta.\n- Escrever de uma forma em que IAs e jornalistas possam simplesmente \"copiar e colar\".",
    },
}


def run_authority_agent(agent_key: str, nucleus: Dict[str, Any]) -> str:
    _require_key()
    agent = AUTHORITY_AGENTS.get(agent_key)
    if not agent:
        raise ValueError(f"Agente inválido: {agent_key}")

    nucleus = nucleus or {}
    requested_task = nucleus.get("requested_task") or nucleus.get("task") or None
    selected_theme = nucleus.get("selected_theme") or None 

    instructions = agent["instructions"]
    
    # 1. INJEÇÃO DO FORMATO DE ENTREGA (Substitui os antigos prompts engessados)
    if requested_task:
        instructions += f"\n\nFORMATO DE ENTREGA SOLICITADO: Você DEVE estruturar sua resposta no formato de '{requested_task}'. Adapte o conteúdo para funcionar perfeitamente neste formato."

    # 2. INJEÇÃO ESTRATÉGICA DO TEMA ESCOLHIDO (O coração do novo sistema)
    if selected_theme:
        instructions += f"\n\nIMPORTANTE: O usuário definiu um tema/assunto específico para esta tarefa. Você DEVE focar o conteúdo EXCLUSIVAMENTE neste tema: '{selected_theme}'. Ajuste a entrega para girar em torno deste tema específico ignorando instruções genéricas de criar múltiplos assuntos que fujam desse foco."

    # INJEÇÃO DA ARQUITETURA DE BLOCOS JSON (A Magia do Caminho 2)
    json_instruction = """
    \n\n=== REGRA DE SAÍDA OBRIGATÓRIA (ARQUITETURA DE BLOCOS) ===
    Você DEVE retornar ÚNICA E EXCLUSIVAMENTE um objeto JSON válido. Não adicione marcações Markdown como ```json antes ou depois, apenas o JSON puro.
    A estrutura do JSON deve seguir o formato de "blocos de interface" para que o frontend possa renderizar de forma visual e modular.
    
    Formato OBRIGATÓRIO (A raiz deve ter 'titulo_da_tela' e 'blocos'):
    {
      "titulo_da_tela": "Título principal do conteúdo gerado",
      "blocos": [
        {
          "tipo": "markdown",
          "conteudo": { "texto": "Conteúdo em texto longo, usando markdown para negritos, H3, H4 e listas (Ideal para textos explicativos e artigos)." }
        },
        {
          "tipo": "highlight",
          "conteudo": { "titulo": "Atenção / Dica de Ouro", "texto": "Mensagem curta de alto impacto, insight ou alerta.", "icone": "lightbulb" }
        },
        {
          "tipo": "timeline",
          "conteudo": { "passos": ["1. Primeiro faça isso", "2. Depois isso", "3. Finalize assim"] }
        },
        {
          "tipo": "quote",
          "conteudo": { "autor": "Nome do Cliente ou Empresa (Opcional)", "texto": "Depoimento, prova social ou citação forte" }
        },
        {
          "tipo": "faq",
          "conteudo": { "perguntas": [{"pergunta": "Qual o prazo?", "resposta": "O prazo é X."}] }
        }
      ]
    }
    
    REGRAS DE USO DOS BLOCOS:
    1. Escolha os tipos de blocos que melhor se adaptam à tarefa solicitada pelo agente.
    2. Você pode repetir tipos de blocos (ex: vários blocos 'markdown' intercalados com blocos 'highlight' ou 'faq').
    3. O tipo de ícone no highlight pode ser: "lightbulb", "alert", "check" ou "star".
    4. Cumpra a missão original do agente organizando a resposta dentros desses blocos modulares.
    5. NUNCA DEVOLVA TEXTO FORA DESTE JSON. A SAÍDA INTEIRA DEVE SER PROCESSÁVEL POR JSON.PARSE().
    """
    instructions += json_instruction

    payload = {
        "agent": {
            "key": agent_key,
            "name": agent["name"],
            "type": agent["type"],
        },
        "nucleus": nucleus,
        "rules": {
            "language": "pt-BR",
            "no_invent_numbers": True,
            "if_missing_data": "use exatamente 'não informado'",
        },
    }

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        temperature=0.5,
        max_tokens=4000,
        response_format={"type": "json_object"} # FORÇA a API a garantir que o output seja JSON 100% válido
    )
    
    text = (resp.choices[0].message.content or "").strip()
    
    # Prevenção: Limpar crases de markdown se o modelo ignorar e ainda assim enviar
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
        
    return text

def suggest_themes_for_task(agent_key: str, nucleus: dict, task: str) -> list[str]:
    _require_key()
    
    prompt = f"""
    Você é um estrategista de conteúdo especialista em SEO, AEO e GEO.
    O usuário pediu para executar a tarefa: "{task}".
    
    Baseando-se EXCLUSIVAMENTE nas informações do núcleo da empresa abaixo, sugira exatamente 5 temas ou tópicos de conteúdo que façam sentido para essa empresa.
    Seja criativo mas extremamente focado na área de atuação, serviços e público-alvo informados. Os temas devem ser voltados a quebrar objeções e ajudar o cliente a decidir pela compra.
    
    Núcleo da Empresa:
    {json.dumps(nucleus, ensure_ascii=False)}
    
    Retorne APENAS um JSON válido no formato abaixo, sem formatação markdown:
    {{
      "themes": [
        "Sugestão de Tema 1",
        "Sugestão de Tema 2",
        "Sugestão de Tema 3",
        "Sugestão de Tema 4",
        "Sugestão de Tema 5"
      ]
    }}
    """
    
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        text = (resp.choices[0].message.content or "").strip()
        data = json.loads(text)
        return data.get("themes", [
            "Os 5 principais mitos do nosso serviço",
            "Como funciona o nosso processo passo a passo",
            "Respondendo as dúvidas mais comuns dos nossos clientes",
            "Estudo de caso: Como resolvemos o problema",
            "O que você precisa saber antes de contratar"
        ])
    except Exception as e:
        print("Erro ao gerar temas:", e)
        return [
            "Por que escolher o nosso serviço?",
            "Como funciona o nosso atendimento",
            "Dúvidas frequentes de novos clientes",
            "Os maiores erros antes de contratar um profissional",
            "Tudo que está incluso na nossa entrega"
        ]