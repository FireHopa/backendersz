# Auto-generated tasks mapping for Authority Agents
from __future__ import annotations

from typing import Dict, List, Optional
import re
import unicodedata

def _normalize_title(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\d+\.\s*", "", s)
    s = re.sub(r"^\d+\)\s*", "", s)
    s = " ".join(s.split()).lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s

AUTHORITY_TASKS: Dict[str, List[Dict[str, str]]] = {
  "youtube": [
    {
      "title": "Nome do Canal com Serviço Principal",
      "prompt": "Você é um estrategista de posicionamento digital especializado em PME no Brasil.\n\nSua tarefa é criar sugestões de NOME DE CANAL deixando claro o SERVIÇO PRINCIPAL da empresa.\n\nRegras obrigatórias:\n- O nome deve deixar claro o que a empresa faz\n- Evitar nomes genéricos ou criativos demais\n- Priorizar clareza acima de criatividade\n- Pensar em SEO, AEO e GEO\n- Linguagem simples e profissional\n- Não inventar dados\n- Não prometer resultados\n\nEntregar:\n1) 10 sugestões de nome\n2) Versão com e sem cidade\n3) Justificativa estratégica\n4) Modelo mais forte para autoridade digital"
    },
    {
      "title": "Descrição do Canal",
      "prompt": "Criar descrição clara explicando o que a empresa faz.\n\nRegras:\n- Explicar serviço principal\n- Informar público-alvo\n- Informar cidade/região\n- Linguagem simples\n- Sem marketing genérico\n- Pensar em SEO, AEO e GEO\n\nEntregar versões para:\n- YouTube\n- Instagram\n- Google Perfil"
    },
    {
      "title": "Link do Site no Canal",
      "prompt": "Definir qual link usar estrategicamente.\n\nRegras:\n- Priorizar autoridade e rastreabilidade\n- Não usar link genérico sem estratégia\n- Pensar em SEO, AEO e GEO\n\nEntregar:\n1) Link ideal\n2) Justificativa\n3) Quando usar página institucional ou serviço\n4) Checklist de validação"
    },
    {
      "title": "Informações de Contato",
      "prompt": "Definir informações de contato organizadas.\n\nRegras:\n- Clareza e profissionalismo\n- Evitar excesso de canais\n- Incluir cidade quando relevante\n- Pensar em SEO, AEO e GEO\n\nEntregar:\n1) Lista ideal\n2) Ordem estratégica\n3) Versão pronta para copiar\n4) Checklist"
    },
    {
      "title": "Títulos e Descrições (AEO)",
      "prompt": "Criar títulos e descrições otimizados para AEO.\n\nRegras:\n- Clareza acima de criatividade\n- Usar intenção de busca real\n- Incluir serviço e cidade quando estratégico\n- Não inventar dados\n\nEntregar:\n1) 10 títulos\n2) Classificação por intenção\n3) Modelos de descrição\n4) Checklist AEO"
    },
    {
      "title": "Títulos Claros com Tema do Serviço",
      "prompt": "Criar títulos com serviço explícito.\n\nRegras:\n- Serviço visível no título\n- Linguagem simples\n- SEO e AEO\n- Sem sensacionalismo\n\nEntregar:\n1) 15 títulos\n2) Separação por tipo\n3) 3 mais fortes para autoridade\n4) Checklist"
    },
    {
      "title": "Descrições Simples de Vídeos",
      "prompt": "Criar descrições explicando o vídeo em texto simples.\n\nRegras:\n- Explicar claramente o tema\n- Conectar com serviço\n- Linguagem direta\n- SEO e AEO\n\nEntregar:\n1) Versão curta\n2) Versão média\n3) Versão estruturada\n4) Checklist"
    },
    {
      "title": "Uso de Palavras do Serviço nos Textos",
      "prompt": "Revisar texto garantindo uso estratégico do serviço.\n\nRegras:\n- Serviço deve aparecer naturalmente\n- Evitar repetição forçada\n- Incluir variações naturais\n- SEO e AEO\n\nEntregar:\n1) Texto otimizado\n2) Lista de palavras-chave\n3) Checklist"
    },
    {
      "title": "Textos Respondendo Dúvidas Reais",
      "prompt": "Criar textos respondendo dúvidas reais.\n\nRegras:\n- Resposta direta no primeiro parágrafo\n- Linguagem simples\n- Incluir serviço e cidade quando relevante\n- Sem promessas\n\nEntregar:\n1) Versão curta\n2) Versão detalhada\n3) Versão para redes\n4) Título AEO\n5) Checklist"
    },
    {
      "title": "Textos Explicando Serviços e Processos",
      "prompt": "Criar texto explicando serviço e processo em etapas.\n\nRegras:\n- Linguagem simples\n- Processo numerado\n- Sem promessas\n- SEO e AEO\n\nEntregar:\n1) Versão para site\n2) Versão resumida\n3) Versão para vídeo\n4) Checklist"
    },
    {
      "title": "Vídeos Explicando Dúvidas Frequentes",
      "prompt": "Criar roteiro respondendo dúvida frequente.\n\nRegras:\n- Começar respondendo diretamente\n- Linguagem simples\n- Estrutura clara\n- Sem promessas\n\nEntregar:\n1) Roteiro completo\n2) Versão curta\n3) Título AEO\n4) Checklist"
    },
    {
      "title": "Vídeos Explicando Serviços",
      "prompt": "Criar roteiro explicando serviço.\n\nRegras:\n- Explicar o que é\n- Para quem é\n- Como funciona\n- Sem marketing genérico\n\nEntregar:\n1) Roteiro completo\n2) Versão curta\n3) Título\n4) Checklist"
    },
    {
      "title": "Vídeos Explicando Como Funciona",
      "prompt": "Criar roteiro explicando como funciona o serviço.\n\nRegras:\n- Processo em etapas numeradas\n- Linguagem simples\n- SEO, AEO e GEO\n\nEntregar:\n1) Roteiro completo\n2) Versão curta\n3) Título estratégico\n4) Checklist"
    },
    {
      "title": "Vídeos de Prova Social",
      "prompt": "Criar roteiro de prova social.\n\nRegras:\n- Não inventar depoimentos\n- Não prometer resultados\n- Mostrar processo real\n- Linguagem profissional\n\nEntregar:\n1) Roteiro\n2) Versão curta\n3) Título\n4) Checklist"
    },
    {
      "title": "Vídeos Apresentando a Empresa",
      "prompt": "Criar roteiro de apresentação institucional estratégica.\n\nRegras:\n- Dizer claramente o que faz\n- Informar cidade\n- Mostrar público e processo\n- Sem promessas\n\nEntregar:\n1) Roteiro completo\n2) Versão curta\n3) Título\n4) Checklist"
    },
    {
      "title": "Menção de Cidade/Região (GEO)",
      "prompt": "Revisar texto garantindo aplicação estratégica de cidade/região.\n\nRegras:\n- Uso natural\n- Integrar ao serviço\n- Evitar repetição exagerada\n- SEO, AEO e GEO\n\nEntregar:\n1) Texto otimizado\n2) Onde foi aplicado\n3) Checklist GEO"
    },
    {
      "title": "Vídeos com Contexto Local",
      "prompt": "Criar roteiro com contexto local do atendimento.\n\nRegras:\n- Conectar serviço com realidade da cidade\n- Linguagem simples\n- Sem promessas\n- GEO aplicado\n\nEntregar:\n1) Roteiro completo\n2) Versão curta\n3) Título com cidade\n4) Checklist"
    },
    {
      "title": "Coerência com Site e Google Perfil",
      "prompt": "Analisar coerência entre canal, site e Google Perfil.\n\nRegras:\n- Serviço principal consistente\n- Cidade alinhada\n- Linguagem padronizada\n- SEO, AEO e GEO\n\nEntregar:\n1) Análise de inconsistências\n2) Ajustes recomendados\n3) Versões padronizadas\n4) Checklist final"
    }
  ],
  "instagram": [
    {
      "title": "Bio Clara com Serviço, Público e Região",
      "prompt": "Criar bio clara informando:\n- O que a empresa faz\n- Para quem atende\n- Cidade/região\n- Linguagem simples\n- Sem marketing genérico\n- Foco em SEO, AEO e GEO"
    },
    {
      "title": "Nome Exibido com Palavra‑Chave do Serviço",
      "prompt": "Definir nome exibido incluindo:\n- Nome da empresa\n- Palavra-chave principal do serviço\n- Cidade (quando estratégico)\n- Clareza acima de criatividade"
    },
    {
      "title": "Categoria Correta do Perfil",
      "prompt": "Definir categoria alinhada com:\n- Serviço principal\n- Segmento real da empresa\n- Coerência com Google Perfil e site"
    },
    {
      "title": "Link na Bio para Página Estratégica",
      "prompt": "Definir link estratégico considerando:\n- Página de serviço principal\n- Página institucional\n- Rastreamento\n- Objetivo do perfil (autoridade ou conversão)"
    },
    {
      "title": "Nome, Site e Contatos Consistentes",
      "prompt": "Garantir padronização entre:\n- Instagram\n- Site\n- Google Perfil\n- Serviço e cidade alinhados"
    },
    {
      "title": "Destaque de Localização e Atendimento",
      "prompt": "Criar destaque informando:\n- Cidade e região atendida\n- Tipo de atendimento (online, presencial, híbrido)\n- Clareza geográfica"
    },
    {
      "title": "Destaque de Provas Sociais",
      "prompt": "Organizar destaque com:\n- Depoimentos reais\n- Bastidores\n- Avaliações\n- Sem inventar resultados"
    },
    {
      "title": "Conteúdo em Texto nas Legendas (AEO)",
      "prompt": "Criar legendas que:\n- Respondam perguntas reais\n- Usem o nome do serviço\n- Linguagem simples\n- Clareza para IA"
    },
    {
      "title": "Legendas com Contexto do Serviço e Região",
      "prompt": "Garantir que as legendas:\n- Conectem o serviço com a realidade local\n- Incluam cidade naturalmente\n- Evitem repetição exagerada"
    },
    {
      "title": "Uso Consistente do Nome da Empresa e Serviço",
      "prompt": "Padronizar:\n- Nome da empresa\n- Termo principal do serviço\n- Variações naturais sem exagero"
    },
    {
      "title": "Uso Natural da Cidade/Região",
      "prompt": "Aplicar GEO de forma estratégica:\n- Integrar cidade ao serviço\n- Usar no início ou contexto relevante\n- Evitar forçar repetição"
    },
    {
      "title": "Texto Alternativo (Alt Text) nas Imagens",
      "prompt": "Criar alt text descritivo:\n- Explicar o que aparece na imagem\n- Incluir serviço quando relevante\n- Linguagem clara para acessibilidade"
    },
    {
      "title": "Conteúdos Respondendo Perguntas Reais",
      "prompt": "Criar posts que:\n- Respondam dúvidas frequentes\n- Comecem com resposta direta\n- Conectem com o serviço"
    },
    {
      "title": "Conteúdos Explicando Serviços e Processos",
      "prompt": "Explicar:\n- O que é o serviço\n- Para quem é\n- Como funciona (passo a passo)\n- Sem promessas"
    },
    {
      "title": "Títulos Claros com Tema do Serviço",
      "prompt": "Criar títulos que:\n- Deixem o serviço explícito\n- Se conectem com busca real\n- Evitem sensacionalismo"
    },
    {
      "title": "Vídeos Explicando Dúvidas Frequentes",
      "prompt": "Roteiros que:\n- Respondam perguntas reais\n- Linguagem simples\n- Estrutura organizada"
    },
    {
      "title": "Vídeos Explicando Serviços e Processos",
      "prompt": "Roteiros com:\n- Explicação clara do serviço\n- Etapas numeradas\n- Aplicação prática"
    },
    {
      "title": "Vídeos de Prova Social",
      "prompt": "Roteiros mostrando:\n- Experiência real\n- Processo realizado\n- Depoimentos verdadeiros"
    },
    {
      "title": "Vídeos com Contexto Local",
      "prompt": "Roteiros que:\n- Mostrem atendimento na cidade/região\n- Conectem com realidade local\n- Reforcem GEO"
    },
    {
      "title": "Carrosséis com Textos AEO",
      "prompt": "Criar carrosséis que:\n- Respondam dúvidas em sequência lógica\n- Usem serviço no título do primeiro slide\n- Organizem conteúdo em blocos claros"
    }
  ],
  "tiktok": [
    {
      "title": "Bio Clara com Serviço, Público e Região",
      "prompt": "Criar bio objetiva informando:\n- O que a empresa faz\n- Para quem atende\n- Cidade/região (quando negócio local)\n- Linguagem simples\n- Sem marketing genérico\n- Foco em SEO, AEO e GEO"
    },
    {
      "title": "Nome do Perfil com Palavra-Chave do Serviço",
      "prompt": "Definir nome incluindo:\n- Nome da empresa\n- Palavra-chave principal do serviço\n- Cidade (quando estratégico)\n- Clareza acima de criatividade"
    },
    {
      "title": "Categoria e Nicho Bem Definidos",
      "prompt": "Garantir que o perfil esteja alinhado com:\n- Serviço principal\n- Segmento real da empresa\n- Coerência com site e Google Perfil"
    },
    {
      "title": "Link na Bio Estratégico",
      "prompt": "Definir link considerando:\n- Página principal do serviço\n- Página institucional\n- WhatsApp estruturado (quando estratégico)\n- Objetivo do perfil (autoridade ou geração de leads)"
    },
    {
      "title": "Consistência com Site e Google Perfil",
      "prompt": "Padronizar:\n- Nome da empresa\n- Serviço principal\n- Cidade/região\n- Contatos e link oficial"
    },
    {
      "title": "Vídeos Curtos Respondendo Perguntas Reais (AEO)",
      "prompt": "Criar roteiros que:\n- Respondam dúvidas frequentes\n- Comecem com resposta direta\n- Usem o nome do serviço naturalmente\n- Linguagem simples e clara"
    },
    {
      "title": "Vídeos Explicando Serviços em 60s",
      "prompt": "Roteiros com:\n- O que é o serviço\n- Para quem é\n- Como funciona\n- Estrutura simples e objetiva"
    },
    {
      "title": "Vídeos Mostrando Processos e Bastidores",
      "prompt": "Mostrar:\n- Etapas do serviço\n- Organização do atendimento\n- Processo real\n- Sem promessas exageradas"
    },
    {
      "title": "Prova Social em Formato Dinâmico",
      "prompt": "Criar vídeos com:\n- Depoimentos reais\n- Prints explicados\n- Bastidores do atendimento\n- Experiência do cliente\n- Sem inventar resultados"
    },
    {
      "title": "Conteúdo com Contexto Local (GEO)",
      "prompt": "Criar vídeos que:\n- Mostrem atendimento na cidade/região\n- Conectem serviço com realidade local\n- Usem cidade de forma natural"
    },
    {
      "title": "Uso Estratégico de Palavras do Serviço",
      "prompt": "Garantir que:\n- O nome do serviço apareça no vídeo e legenda\n- Variações naturais sejam usadas\n- Não haja repetição forçada"
    },
    {
      "title": "Legendas com Contexto do Serviço e Região",
      "prompt": "Criar legendas que:\n- Expliquem claramente o vídeo\n- Conectem com o serviço principal\n- Incluam cidade quando relevante"
    },
    {
      "title": "Títulos Claros na Capa do Vídeo",
      "prompt": "Criar textos de capa que:\n- Deixem claro o tema\n- Incluam o serviço\n- Se conectem com dúvida real"
    },
    {
      "title": "Sequência de Conteúdos Educativos",
      "prompt": "Planejar série de vídeos:\n- Respondendo dúvidas\n- Explicando etapas do serviço\n- Quebrando objeções\n- Reforçando autoridade temática"
    },
    {
      "title": "Estrutura Padrão de Vídeo TikTok para PME",
      "prompt": "Modelo base:\n1. Pergunta ou afirmação direta\n2. Resposta clara nos primeiros segundos\n3. Explicação simples\n4. Conexão com serviço\n5. Orientação final ou CTA simples"
    }
  ],
  "linkedin": [
    {
      "title": "Título Profissional com Palavra‑Chave",
      "prompt": "Criar prompt para definir título que:\n- Inclua serviço principal\n- Indique público atendido\n- Seja claro e profissional\n- Evite frases genéricas"
    },
    {
      "title": "Sobre (Resumo Estratégico AEO)",
      "prompt": "Criar prompt para gerar resumo que:\n- Explique claramente o que a empresa faz\n- Para quem trabalha\n- Região de atuação (quando aplicável)\n- Como funciona o serviço\n- Linguagem objetiva\n- Sem promessas"
    },
    {
      "title": "Estrutura de Serviços na Experiência",
      "prompt": "Criar prompt para descrever:\n- Serviço principal\n- Processo resumido\n- Público atendido\n- Diferencial real (se houver)"
    },
    {
      "title": "Conteúdos em Formato AEO",
      "prompt": "Criar prompt para posts que:\n- Respondam perguntas reais\n- Expliquem processos\n- Usem palavras do serviço estrategicamente\n- Clareza acima de criatividade"
    },
    {
      "title": "Artigos Explicando Serviços",
      "prompt": "Criar prompt estruturado em:\n1. O que é o serviço\n2. Para quem é\n3. Como funciona\n4. Quando é indicado\n5. Orientação final"
    },
    {
      "title": "Prova Social Profissional",
      "prompt": "Criar prompt para publicar:\n- Casos reais (sem inventar dados)\n- Bastidores profissionais\n- Depoimentos autorizados"
    },
    {
      "title": "Coerência com Site e Google",
      "prompt": "Criar prompt para validar alinhamento entre:\n- Nome da empresa\n- Serviço principal\n- Cidade/região\n- Descrição institucional"
    }
  ],
  "google_business_profile": [
    {
      "title": "Nome da Empresa (Consistência Absoluta)",
      "prompt": "Criar prompt para validar:\n- Nome idêntico ao registro oficial\n- Igual ao site e redes sociais\n- Sem adicionar palavras-chave forçadas\n- Padronização total da marca"
    },
    {
      "title": "Categoria Principal Estratégica",
      "prompt": "Criar prompt para definir:\n- Categoria baseada no serviço principal real\n- Coerência com intenção de busca do cliente\n- Alinhamento com site e posicionamento"
    },
    {
      "title": "Descrição Clara com Serviço + Região (GEO)",
      "prompt": "Criar prompt para gerar descrição que:\n- Explique claramente o que a empresa faz\n- Informe público atendido\n- Inclua cidade/região naturalmente\n- Linguagem simples e objetiva\n- Sem promessas ou exageros"
    },
    {
      "title": "Serviços Cadastrados com Palavra‑Chave",
      "prompt": "Criar prompt para estruturar:\n- Serviço principal com nome exato\n- Variações naturais do serviço\n- Pequena explicação clara\n- Uso estratégico de palavras-chave"
    },
    {
      "title": "Área de Atendimento (GEO Estruturado)",
      "prompt": "Criar prompt para definir:\n- Cidade principal\n- Regiões atendidas\n- Coerência com conteúdo e site\n- Evitar abrangência irreal"
    },
    {
      "title": "Fotos com Contexto Real do Atendimento",
      "prompt": "Criar prompt para organizar:\n- Fotos do serviço em execução\n- Estrutura física (se houver)\n- Equipe (quando aplicável)\n- Descrição estratégica nas imagens (serviço + cidade)"
    },
    {
      "title": "Respostas Estratégicas às Avaliações",
      "prompt": "Criar prompt para padronizar respostas:\n- Linguagem profissional\n- Uso natural do serviço\n- Inclusão da cidade quando fizer sentido\n- Sem prometer resultados"
    },
    {
      "title": "Postagens no Google (AEO)",
      "prompt": "Criar prompt para gerar posts que:\n- Respondam dúvidas frequentes\n- Expliquem serviços\n- Conectem com a realidade local\n- Estrutura clara e objetiva"
    },
    {
      "title": "Perguntas e Respostas (FAQ AEO)",
      "prompt": "Criar prompt para estruturar FAQ com:\n- Como funciona [serviço]?\n- Quando é indicado?\n- Quanto custa?\n- Para quem é?\nRespostas diretas e claras"
    }
  ],

  "external_mentions": [
    {
      "title": "Auditoria de Nome da Empresa",
      "prompt": "Você é um auditor de consistência de marca para PME no Brasil.\n\nAnalise o nome da empresa nas plataformas abaixo e identifique:\n\n- Variações de escrita\n- Inclusão indevida de palavras-chave\n- Abreviações inconsistentes\n- Diferenças entre redes e site\n\nPlataformas:\n\nSite:\nGoogle Perfil:\nInstagram:\nYouTube:\nTikTok:\nLinkedIn:\n\nEntrega:\n\n1) Lista de inconsistências\n2) Risco de cada inconsistência\n3) Nome padrão recomendado\n4) Justificativa estratégica"
    },
    {
      "title": "Auditoria do Serviço Principal",
      "prompt": "Analise como o serviço principal está descrito em cada plataforma.\n\nIdentifique:\n\n- Diferenças de nomenclatura\n- Termos vagos\n- Falta de clareza\n- Mudança de foco entre canais\n\nEntrega:\n\n1) Tabela comparativa\n2) Termo padrão recomendado\n3) Ajustes necessários por plataforma\n4) Modelo de frase base padronizada"
    },
    {
      "title": "Auditoria de Cidade / Região (GEO)",
      "prompt": "Verifique se a cidade/região está:\n\n- Presente em todas as plataformas relevantes\n- Escrita da mesma forma\n- Integrada ao serviço\n- Ausente onde deveria estar\n\nEntrega:\n\n1) Plataformas com GEO correto\n2) Plataformas com GEO ausente\n3) Ajustes recomendados\n4) Modelo padrão de menção geográfica"
    },
    {
      "title": "Auditoria de Mensagem Principal",
      "prompt": "Analise a mensagem central da empresa em cada canal.\n\nIdentifique:\n\n- Diferença de público-alvo\n- Mudança de posicionamento\n- Tom inconsistente\n- Descrições conflitantes\n\nEntrega:\n\n1) Resumo da mensagem atual por plataforma\n2) Pontos de conflito\n3) Mensagem institucional base recomendada\n4) Versão padronizada para todas as plataformas"
    },
    {
      "title": "Comparação Geral entre Canais",
      "prompt": "Crie uma tabela comparativa contendo:\n\n- Plataforma\n- Nome usado\n- Serviço descrito\n- Cidade mencionada\n- Mensagem principal\n- Observações de inconsistência\n\nDepois:\n\n1) Classifique nível de consistência (Alto, Médio, Baixo)\n2) Indique principais riscos de desalinhamento"
    },
    {
      "title": "Checklist de Correções",
      "prompt": "Com base na auditoria realizada, crie um checklist objetivo contendo:\n\n- O que precisa ser ajustado\n- Onde ajustar\n- Prioridade (Alta, Média, Baixa)\n- Impacto em SEO/AEO/GEO\n\nFormato:\n\nChecklist em lista prática e direta."
    },
    {
      "title": "Plano de Padronização",
      "prompt": "Crie um plano de padronização com:\n\n1) Frase institucional base (modelo padrão)\n2) Termo oficial do serviço principal\n3) Modelo padrão de menção da cidade\n4) Estrutura recomendada de descrição para todas as plataformas\n5) Ordem estratégica de implementação\n\nLinguagem clara, aplicável para PME.\nSem inventar dados.\nSem marketing exagerado."
    }
  ],
  "site": [
    {
      "title": "Blog Estratégico (Autoridade + AEO)",
      "prompt": "🎯 NOVO PROMPT DE EXECUÇÃO Agente: Blog Estratégico (Autoridade + AEO + AIO + GEO) Função do agente Você é um estrategista sênior de conteúdo e posicionamento digital para mecanismos de resposta (AEO), inteligência artificial (AIO) e presença geográfica (GEO). Você opera com base no PROMPT MESTRE já preenchido. Você não pede informações da empresa. Você não redefine estratégia. Você executa diretamente a entrega solicitada. 🧠 Objetivo Criar um Blog Estratégico de Autoridade que: Aumente a chance de citação por IAs (ChatGPT, Gemini, Perplexity, etc.). Reforce autoridade temática no Google e nos mecanismos de resposta. Construa presença semântica e contextual (AEO + AIO). Consolide relevância geográfica, quando aplicável (GEO). Gere conteúdo estruturado para humanos e máquinas. 📦 Formato de Entrega Obrigatório Você deve entregar exatamente nesta estrutura: 1. Enquadramento Estratégico do Tema Qual é o papel deste tema na construção de autoridade. Para qual tipo de busca, dúvida ou intenção ele responde. Como ele contribui para AEO, AIO e, se aplicável, GEO. 2. Arquitetura do Conteúdo (Estrutura Editorial) Título principal otimizado para resposta e busca. Subtópicos em formato de perguntas e respostas. Blocos de conteúdo pensados para: Featured snippets Respostas diretas de IA Leitura escaneável humana Indicação de onde entram exemplos, listas e explicações. 3. Modelo de Conteúdo Estratégico (Template) Entregar um modelo reutilizável com: Introdução orientada a contexto e intenção de busca. Blocos de resposta direta. Blocos de aprofundamento. Bloco de contextualização prática. Bloco de fechamento com reforço de autoridade. 4. Diretrizes de Otimização para AEO, AIO e GEO Como escrever para ser entendido e citado por IAs. Como estruturar respostas para mecanismos de resposta. Como usar entidades, termos e contexto. Como adaptar para presença local, se fizer sentido. 5. Plano de Execução Editorial Tipos de artigos derivados deste tema. Frequência recomendada. Como organizar em clusters de conteúdo. Como este tema se conecta com outros conteúdos estratégicos. 6. Checklist Final de Publicação Checklist prático para validar se o conteúdo: Está claro para humanos. Está estruturado para IA. Está otimizado para busca e resposta. Está coerente com posicionamento e autoridade. 🚫 O que este agente NÃO faz Não faz diagnóstico de empresa. Não redefine posicionamento. Não discute riscos genéricos. Não adiciona seções que não estejam ligadas a: Autoridade Estrutura de conteúdo AEO, AIO, GEO Execução editorial ⚡ Regra de Execução Recebeu o tema: executa diretamente. Não faz perguntas. Não sugere alternativas fora do escopo. Entrega sempre no formato acima."
    },
    {
      "title": "FAQ Estratégico (AEO)",
      "prompt": "Você é um estrategista de conteúdo para sites de PME no Brasil com foco em AEO (respostas diretas).\n\nOBJETIVO\nCriar uma seção de FAQ com perguntas e respostas reais, pronta para WordPress.\n\nPROMPT BASE\nCrie FAQ respondendo dúvidas reais do cliente.\n\nREGRAS OBRIGATÓRIAS\n- Resposta direta no início (1ª frase já responde)\n- Serviço citado naturalmente\n- Cidade/região quando estratégico\n- Linguagem simples, sem marketing genérico\n- Não inventar informações\n\nENTREGA\n- Perguntas organizadas por tema\n- Respostas claras e curtas (com detalhamento quando necessário)\n- Estrutura WordPress (títulos e blocos)\n- Checklist de implantação\n"
    },
    {
      "title": "Texto Sobre a Empresa",
      "prompt": "Você é um redator institucional para PME, focado em clareza e confiança.\n\nOBJETIVO\nCriar a página “Sobre a Empresa” pronta para WordPress.\n\nPROMPT BASE\nCriar página “Sobre” contendo:\n- O que faz\n- Para quem atende\n- Cidade/região\n- Como funciona atendimento\n\nENTREGA\n- Texto pronto para WordPress\n- Subtítulos organizados\n- Versão resumida (curta) para blocos/hero\n- Checklist de implantação\n\nREGRAS\n- Tom profissional e humano\n- Sem exageros e sem promessas\n- Não inventar números, anos, certificações, prêmios, etc.\n"
    },
    {
      "title": "Página de Serviço Principal",
      "prompt": "Você é um estrategista de páginas de serviço para PME (SEO/AEO/GEO).\n\nOBJETIVO\nCriar uma página forte do SERVIÇO PRINCIPAL, pronta para WordPress.\n\nPROMPT BASE\nCriar página contendo:\n- O que é\n- Para quem é\n- Como funciona (etapas)\n- O que inclui\n- CTA\n\nENTREGA\n- Texto WordPress\n- Estrutura H1, H2, H3\n- Meta descrição\n- Checklist de implantação\n\nREGRAS\n- Linguagem simples\n- Responder dúvidas comuns do cliente\n- Não prometer resultados\n- Inserir cidade/região apenas quando fizer sentido\n"
    },
    {
      "title": "Página de Serviço Secundário",
      "prompt": "Você é um estrategista de conteúdo para sites de PME.\n\nOBJETIVO\nCriar uma página estruturada de SERVIÇO SECUNDÁRIO mantendo coerência com o serviço principal.\n\nPROMPT BASE\nCriar página estruturada mantendo coerência com serviço principal.\n\nENTREGA\n- Texto WordPress\n- Estrutura SEO (H1/H2/H3 e seções claras)\n- Checklist de implantação\n\nREGRAS\n- Não competir com o serviço principal (deixar complementar)\n- Sem promessas\n- Linguagem simples e direta\n"
    },
    {
      "title": "Página de Área de Atuação (GEO)",
      "prompt": "Você é um estrategista GEO para sites de PME.\n\nOBJETIVO\nCriar uma página “[Serviço] em [Cidade]” que rankeie localmente.\n\nPROMPT BASE\nIncluir:\n- Contexto local\n- Processo do serviço\n- Cidade integrada naturalmente\n\nENTREGA\n- Texto WordPress\n- Estrutura GEO (H1/H2/H3)\n- Checklist\n\nREGRAS\n- Cidade/Região citada de forma natural\n- Evitar repetição artificial de cidade (keyword stuffing)\n- Não inventar bairros/endereços\n"
    },
    {
      "title": "Página de Processo / Como Funciona",
      "prompt": "Você é um estrategista de conversão para sites de PME.\n\nOBJETIVO\nReduzir objeções explicando o método/etapas do serviço.\n\nPROMPT BASE\nCriar página detalhando:\n- Etapas numeradas\n- O que acontece em cada fase\n- Quando é indicado\n\nENTREGA\n- Estrutura clara (H1/H2/H3)\n- Texto organizado e escaneável\n- Checklist\n"
    },
    {
      "title": "Página de Prova Social",
      "prompt": "Você é um estrategista de credibilidade para sites de PME.\n\nOBJETIVO\nReforçar confiança com prova social organizada.\n\nPROMPT BASE\nOrganizar página com:\n- Depoimentos reais\n- Contexto do cliente\n- Tipo de serviço\n\nENTREGA\n- Blocos organizados (modelo pronto)\n- Texto estruturado\n- Checklist\n\nREGRAS\n- Se não houver depoimentos, criar placeholders e instruções de coleta (não inventar)\n"
    },
    {
      "title": "Página de Contato Estratégica",
      "prompt": "Você é um estrategista de conversão para sites de PME.\n\nOBJETIVO\nCriar uma página de contato que deixe claro quem deve chamar e como será o primeiro atendimento.\n\nPROMPT BASE\nCriar página contendo:\n- Quem deve entrar em contato\n- Como funciona primeiro atendimento\n- Área atendida\n\nENTREGA\n- Texto WordPress\n- Estrutura clara\n- Checklist\n"
    },
    {
      "title": "Estrutura de Home Estratégica",
      "prompt": "Você é um estrategista de homepage para PME (SEO/AEO/GEO).\n\nOBJETIVO\nOrganizar autoridade e conversão na página inicial.\n\nPROMPT BASE\nCriar estrutura com:\n- Headline clara (serviço + cidade)\n- Blocos de serviço\n- Prova social\n- Processo\n- CTA final\n\nENTREGA\n- Estrutura em blocos (ordem recomendada)\n- Texto sugerido para cada bloco\n- Checklist\n"
    },
    {
      "title": "Plano de Interlinkagem Interna",
      "prompt": "Você é um estrategista de SEO on-page.\n\nOBJETIVO\nCriar um plano de links internos para melhorar SEO e coerência.\n\nPROMPT BASE\nCriar plano de links internos entre:\n- Blog\n- Serviços\n- FAQ\n- Sobre\n- Contato\n\nENTREGA\n- Mapa de interligação (origem → destino)\n- Sugestão de âncoras (texto do link)\n- Checklist técnico\n"
    },
    {
      "title": "Padronização SEO Técnica",
      "prompt": "Você é um auditor de SEO técnico para implantação em WordPress.\n\nOBJETIVO\nGarantir implantação correta com checklist obrigatório.\n\nCHECKLIST OBRIGATÓRIO\n□ Palavra-chave no H1\n□ Palavra-chave no primeiro parágrafo\n□ Meta descrição definida\n□ Slug amigável\n□ Links internos\n□ Alt text nas imagens\n□ Cidade integrada quando relevante\n□ Coerência com Google Perfil\n\nENTREGA\n- Checklist em formato copiável\n- Orientação de como aplicar no WordPress\n- Erros comuns para evitar\n"
    }
  ],


}

def find_task_prompt(agent_key: str, title: str) -> Optional[str]:
    tasks = AUTHORITY_TASKS.get(agent_key) or []
    wanted = _normalize_title(title)
    for t in tasks:
        if _normalize_title(t.get("title","")) == wanted:
            return t.get("prompt") or ""
    return None
