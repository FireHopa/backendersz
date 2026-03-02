from __future__ import annotations
import json
import uuid
import hashlib
import io
import re
try:
    import pypdf
except ImportError:
    pypdf = None
try:
    import docx
except ImportError:
    docx = None
    
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from .db import init_db, get_session
from .models import Robot, ChatMessage, CompetitionAnalysis, AuthorityEdit, AuthorityAgentRun, BusinessCore, User
from .schemas import (
    BriefingIn,
    RobotOut,
    RobotDetail,
    RobotUpdateIn,
    ChatIn,
    ChatMessageOut,
    MessageUpdateIn,
    AuthorityAssistantIn,
    AuthorityAssistantOut,
    AuthorityEditOut,
    AuthorityAgentRunIn,
    AuthorityAgentHistoryOut,
    AuthorityAgentRunOut,
    CompetitionFindRequest,
    CompetitionAnalyzeRequest,
    CompetitionJobV2Out,
    CompetitionReportV2Out,
    CompetitionFindOut,
)
from .ai import build_robot_from_briefing, chat_with_robot, transcribe_audio, find_competitors, build_competition_result, authority_assistant, run_authority_agent

# Importações de Autenticação
from .deps import get_current_user
from .auth import router as auth_router
from pydantic import BaseModel

class SuggestThemesRequest(BaseModel):
    agent_key: str
    task: str
    nucleus: dict

app = FastAPI(title="Authority Robot Panel API")

# Registando as rotas de Auth
app.include_router(auth_router)

async def extract_text_from_file(file: UploadFile) -> str:
    """Extrai texto de PDFs, DOCX ou ficheiros de texto puro."""
    content = await file.read()
    filename = file.filename.lower()
    text = ""

    if filename.endswith(".pdf"):
        if not pypdf:
            raise HTTPException(status_code=500, detail="pypdf não instalado no backend.")
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao ler o PDF: {str(e)}")
            
    elif filename.endswith(".docx"):
        if not docx:
            raise HTTPException(status_code=500, detail="python-docx não instalado.")
        try:
            doc = docx.Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao ler o DOCX: {str(e)}")
            
    elif filename.endswith(".txt") or filename.endswith(".md") or filename.endswith(".csv"):
        try:
            text = content.decode("utf-8")
        except:
            text = content.decode("latin-1", errors="ignore")
    else:
        raise HTTPException(status_code=400, detail="Formato não suportado. Utilize PDF, DOCX, TXT ou MD.")

    return text.strip()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/api/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

# --- ROBOT HELPERS (Atualizado para Isolamento Total) ---
def _get_robot_or_404(public_id: str, session: Session, current_user: User) -> Robot:
    if public_id == "business-core-global":
        # O Núcleo Global continua a ser acessível a todos os utilizadores autenticados (para leitura)
        robot = session.exec(select(Robot).where(Robot.public_id == public_id)).first()
        if not robot:
            robot = Robot(
                public_id="business-core-global",
                title="[SISTEMA] Núcleo Global",
                description="Armazena os ficheiros do Núcleo da Empresa.",
                system_instructions="Não usado diretamente no chat."
            )
            session.add(robot)
            session.commit()
            session.refresh(robot)
        return robot
        
    # Filtra rigorosamente pelo user_id do utilizador atual
    robot = session.exec(select(Robot).where(Robot.public_id == public_id, Robot.user_id == current_user.id)).first()
    if not robot:
        raise HTTPException(status_code=404, detail="Assistente não encontrado ou não tem permissão para aceder.")
    return robot

# --- ROBOT ROUTES ---
@app.get("/api/robots", response_model=list[RobotOut])
def list_robots(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robots = session.exec(
        select(Robot)
        .where(Robot.public_id != "business-core-global")
        .where(Robot.user_id == current_user.id)  # Apenas os robôs do utilizador atual
        .order_by(Robot.created_at.desc())
    ).all()
    return [
        RobotOut(
            public_id=r.public_id,
            title=r.title,
            description=r.description or "",
            avatar_data=r.avatar_data,
            created_at=r.created_at.isoformat(),
        )
        for r in robots
    ]

@app.get("/api/robots/{public_id}", response_model=RobotDetail)
def get_robot(public_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    return RobotDetail(
        public_id=robot.public_id,
        title=robot.title,
        description=robot.description or "",
        avatar_data=robot.avatar_data,
        system_instructions=robot.system_instructions,
        created_at=robot.created_at.isoformat(),
        knowledge_files_json=robot.knowledge_files_json
    )

@app.delete("/api/robots/{public_id}")
def delete_robot(public_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    msgs = session.exec(select(ChatMessage).where(ChatMessage.robot_id == robot.id)).all()
    for m in msgs:
        session.delete(m)
    session.delete(robot)
    session.commit()
    return {"ok": True}

@app.post("/api/robots", response_model=RobotOut)
def create_robot(brief: BriefingIn, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    try:
        built = build_robot_from_briefing(brief.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    robot = Robot(
        user_id=current_user.id,  # Associa o robô ao utilizador atual
        public_id=uuid.uuid4().hex,
        title=built["title"],
        description=built.get("description") or "",
        avatar_data=None,
        system_instructions=built["system_instructions"],
    )
    session.add(robot)
    session.commit()
    session.refresh(robot)

    return RobotOut(
        public_id=robot.public_id,
        title=robot.title,
        description=robot.description or "",
        avatar_data=robot.avatar_data,
        created_at=robot.created_at.isoformat(),
    )

@app.patch("/api/robots/{public_id}", response_model=RobotDetail)
def update_robot(public_id: str, body: RobotUpdateIn, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    data = body.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(robot, k, v)
    session.add(robot)
    session.commit()
    session.refresh(robot)
    return RobotDetail(
        public_id=robot.public_id,
        title=robot.title,
        description=robot.description or "",
        avatar_data=robot.avatar_data,
        system_instructions=robot.system_instructions,
        created_at=robot.created_at.isoformat(),
        knowledge_files_json=robot.knowledge_files_json
    )

# --- BUSINESS CORE ROUTES ---
@app.get("/api/robots/{public_id}/business-core")
def get_business_core(public_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    core = session.exec(select(BusinessCore).where(BusinessCore.robot_id == robot.id)).first()
    
    if not core:
        core = BusinessCore(robot_id=robot.id)
        session.add(core)
        session.commit()
        session.refresh(core)
        
    return core

@app.patch("/api/robots/{public_id}/business-core")
def update_business_core(public_id: str, payload: dict, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    core = session.exec(select(BusinessCore).where(BusinessCore.robot_id == robot.id)).first()
    
    if not core:
        core = BusinessCore(robot_id=robot.id)
        session.add(core)
        
    for k, v in payload.items():
        if hasattr(core, k) and v is not None and k not in ("id", "robot_id"):
            setattr(core, k, v)
            
    core.updated_at = datetime.utcnow()
    session.add(core)
    session.commit()
    session.refresh(core)
    return core

# --- MESSAGES ROUTES ---
@app.get("/api/robots/{public_id}/messages", response_model=list[ChatMessageOut])
def list_messages(public_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    msgs = session.exec(
        select(ChatMessage)
        .where(ChatMessage.robot_id == robot.id)
        .order_by(ChatMessage.created_at.asc())
    ).all()
    return [
        ChatMessageOut(id=m.id, role=m.role, content=m.content, created_at=m.created_at.isoformat())
        for m in msgs
    ]

@app.post("/api/robots/{public_id}/authority-assistant", response_model=AuthorityAssistantOut)
def authority_assistant_route(public_id: str, body: AuthorityAssistantIn, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)

    edits = session.exec(
        select(AuthorityEdit)
        .where(AuthorityEdit.robot_id == robot.id)
        .order_by(AuthorityEdit.created_at.desc())
    ).all()
    edits_history = []
    for e in edits[:30]:
        try:
            changes = json.loads(e.changes_json or "[]")
        except Exception:
            changes = []
        edits_history.append(
            {
                "id": e.id,
                "created_at": e.created_at.isoformat(),
                "user_message": e.user_message,
                "summary": e.summary or "",
                "changes_made": changes,
                "before_score": e.before_score,
                "after_score": e.after_score,
            }
        )

    before_instructions = robot.system_instructions or ""
    before_hash = hashlib.sha256(before_instructions.encode("utf-8")).hexdigest()

    result = authority_assistant(
        robot_system_instructions=before_instructions,
        user_message=body.message,
        history=body.history or [],
        authority_edits_history=edits_history,
    )

    if result.get("apply_change") and result.get("updated_system_instructions"):
        updated = str(result["updated_system_instructions"])
        after_hash = hashlib.sha256(updated.encode("utf-8")).hexdigest()

        exists = session.exec(
            select(AuthorityEdit).where(
                AuthorityEdit.robot_id == robot.id,
                AuthorityEdit.after_hash == after_hash,
            )
        ).first()

        if not exists:
            robot.system_instructions = updated
            session.add(robot)
            session.commit()
            session.refresh(robot)

            changes_made = result.get("changes_made") or []
            summary = ""
            if isinstance(changes_made, list) and changes_made:
                summary = "; ".join(
                    [str(c.get("title") or c.get("change") or c.get("what") or "").strip() for c in changes_made]
                )[:280]

            edit = AuthorityEdit(
                robot_id=robot.id,
                user_message=str(body.message),
                assistant_reply=str(result.get("assistant_reply") or ""),
                changes_json=json.dumps(changes_made, ensure_ascii=False),
                summary=summary,
                before_score=int(result.get("before_score") or 0),
                after_score=int(result.get("after_score") or 0),
                before_hash=before_hash,
                after_hash=after_hash,
            )
            session.add(edit)
            session.commit()

    return AuthorityAssistantOut(
        apply_change=bool(result.get("apply_change") or False),
        before_score=int(result.get("before_score") or 0),
        after_score=int(result.get("after_score") or 0),
        criteria=list(result.get("criteria") or []),
        changes_made=list(result.get("changes_made") or []),
        suggestions=list(result.get("suggestions") or []),
        updated_system_instructions=result.get("updated_system_instructions"),
        assistant_reply=str(result.get("assistant_reply") or ""),
    )

@app.get("/api/robots/{public_id}/authority-edits", response_model=list[AuthorityEditOut])
def list_authority_edits(public_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    edits = session.exec(
        select(AuthorityEdit)
        .where(AuthorityEdit.robot_id == robot.id)
        .order_by(AuthorityEdit.created_at.desc())
    ).all()
    out: list[AuthorityEditOut] = []
    for e in edits:
        try:
            changes = json.loads(e.changes_json or "[]")
        except Exception:
            changes = []
        out.append(
            AuthorityEditOut(
                id=e.id,
                created_at=e.created_at.isoformat(),
                user_message=e.user_message,
                summary=e.summary or "",
                changes_made=changes,
                before_score=e.before_score,
                after_score=e.after_score,
            )
        )
    return out

@app.delete("/api/robots/{public_id}/messages")
def clear_messages(public_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    msgs = session.exec(select(ChatMessage).where(ChatMessage.robot_id == robot.id)).all()
    for m in msgs:
        session.delete(m)
    session.commit()
    return {"ok": True}

@app.patch("/api/robots/{public_id}/messages/{message_id}", response_model=ChatMessageOut)
def update_message(public_id: str, message_id: int, body: MessageUpdateIn, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    msg = session.exec(
        select(ChatMessage).where(ChatMessage.id == message_id, ChatMessage.robot_id == robot.id)
    ).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Mensagem não encontrada")
    if msg.role != "user":
        raise HTTPException(status_code=400, detail="Só é permitido editar mensagens do utilizador")

    msg.content = body.content
    session.add(msg)
    session.commit()
    session.refresh(msg)
    return ChatMessageOut(id=msg.id, role=msg.role, content=msg.content, created_at=msg.created_at.isoformat())

@app.post("/api/robots/{public_id}/audio", response_model=ChatMessageOut)
async def chat_audio(public_id: str, file: UploadFile = File(...), session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    audio_bytes = await file.read()
    try:
        text = transcribe_audio(audio_bytes, filename=file.filename or "audio.webm")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao transcrever o áudio: {e}")

    msgs = session.exec(
        select(ChatMessage)
        .where(ChatMessage.robot_id == robot.id)
        .order_by(ChatMessage.created_at.asc())
    ).all()
    history = [{"role": m.role, "content": m.content} for m in msgs][-20:]

    user_msg = ChatMessage(robot_id=robot.id, role="user", content=text)
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)

    try:
        answer = chat_with_robot(robot.system_instructions, history, text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    assistant_msg = ChatMessage(robot_id=robot.id, role="assistant", content=answer)
    session.add(assistant_msg)
    session.commit()
    session.refresh(assistant_msg)

    return ChatMessageOut(
        id=assistant_msg.id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        created_at=assistant_msg.created_at.isoformat(),
    )

@app.post("/api/robots/{public_id}/chat", response_model=ChatMessageOut)
def chat(public_id: str, body: ChatIn, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    robot = _get_robot_or_404(public_id, session, current_user)
    msgs = session.exec(
        select(ChatMessage)
        .where(ChatMessage.robot_id == robot.id)
        .order_by(ChatMessage.created_at.asc())
    ).all()
    history = [{"role": m.role, "content": m.content} for m in msgs][-20:]

    user_msg = ChatMessage(robot_id=robot.id, role="user", content=body.message)
    session.add(user_msg)
    session.commit()
    session.refresh(user_msg)

    try:
        answer = chat_with_robot(robot.system_instructions, history, body.message, use_web=body.use_web, web_max_results=body.web_max_results, web_allowed_domains=body.web_allowed_domains)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    assistant_msg = ChatMessage(robot_id=robot.id, role="assistant", content=answer)
    session.add(assistant_msg)
    session.commit()
    session.refresh(assistant_msg)

    return ChatMessageOut(
        id=assistant_msg.id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        created_at=assistant_msg.created_at.isoformat(),
    )

# --- COMPETITION HELPERS ---
def _update_analysis(session, obj, **kwargs):
    for k, v in kwargs.items():
        setattr(obj, k, v)
    obj.updated_at = datetime.utcnow()
    session.add(obj)
    session.commit()
    session.refresh(obj)

def _domain(url: str) -> str:
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc
    except:
        return url

def _run_analysis_job(public_id: str):
    from sqlmodel import Session
    from .db import engine
    import json
    from .ai import build_competition_result
    
    with Session(engine) as session:
        obj = None
        try:
            obj = session.exec(select(CompetitionAnalysis).where(CompetitionAnalysis.public_id == public_id)).first()
            if not obj:
                return
            
            _update_analysis(session, obj, status="running", stage="Coletando sinais públicos", progress=0.15)
            
            instagrams = json.loads(obj.instagrams_json or "[]")
            sites = json.loads(obj.sites_json or "[]")
            company = json.loads(obj.company_json or "{}") if obj.company_json else {}

            competitors = []
            for s in sites:
                competitors.append({"name": _domain(s), "website_url": s})
            
            _update_analysis(session, obj, stage="Analisando presença digital", progress=0.40)
            
            _update_analysis(session, obj, stage="Consolidando inteligência", progress=0.75)

            result = build_competition_result(company=company, competitors=competitors[:3])

            _update_analysis(session, obj, status="done", stage="Concluído", progress=1.0, result_json=json.dumps(result, ensure_ascii=False))

        except Exception as e:
            if obj:
                _update_analysis(session, obj, status="error", stage="Erro no processamento", progress=1.0, error=str(e))
            print(f"Job Error: {e}")

# --- COMPETITION ROUTES (V2 ONLY) ---
@app.post("/api/competition/find-competitors", response_model=CompetitionFindOut)
def competition_find_competitors_v2(payload: CompetitionFindRequest, current_user: User = Depends(get_current_user)):
    briefing = payload.briefing.model_dump()
    mapped = {
        "company_name": briefing.get("nome_empresa"),
        "niche": briefing.get("segmento"),
        "region": briefing.get("cidade_estado"),
        "services": briefing.get("servicos"),
        "audience": briefing.get("publico_alvo"),
        "offer": briefing.get("servicos"),
    }
    data = find_competitors(mapped)
    return data

@app.post("/api/competition/analyze", response_model=CompetitionJobV2Out)
def competition_analyze_v2(payload: CompetitionAnalyzeRequest, bg: BackgroundTasks, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    public_id = str(uuid.uuid4())
    instas = payload.instagrams or []
    sites = payload.sites or []
    briefing = payload.briefing.model_dump() if payload.briefing else None

    obj = CompetitionAnalysis(
        user_id=current_user.id,  # Associa a análise ao utilizador logado
        public_id=public_id,
        instagrams_json=json.dumps(instas, ensure_ascii=False),
        sites_json=json.dumps(sites, ensure_ascii=False),
        company_json=json.dumps(briefing or {}, ensure_ascii=False),
        status="queued",
        stage="Na fila",
        progress=0.0,
    )
    session.add(obj)
    session.commit()
    session.refresh(obj)

    bg.add_task(_run_analysis_job, obj.public_id)

    return CompetitionJobV2Out(
        job_id=obj.public_id,
        report_id=obj.public_id,
        status=obj.status,
        stage=obj.stage,
        progress=obj.progress,
    )

@app.get("/api/competition/jobs/{job_id}", response_model=CompetitionJobV2Out)
def competition_job_v2(job_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    obj = session.exec(select(CompetitionAnalysis).where(CompetitionAnalysis.public_id == job_id, CompetitionAnalysis.user_id == current_user.id)).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return CompetitionJobV2Out(
        job_id=obj.public_id,
        report_id=obj.public_id,
        status=obj.status,
        stage=obj.stage,
        progress=obj.progress,
        error=obj.error,
        warning=None
    )

@app.get("/api/competition/reports/{report_id}", response_model=CompetitionReportV2Out)
def competition_report_v2(report_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    obj = session.exec(select(CompetitionAnalysis).where(CompetitionAnalysis.public_id == report_id, CompetitionAnalysis.user_id == current_user.id)).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Relatório não encontrado")
    
    if obj.status == "error":
         raise HTTPException(status_code=400, detail=f"Erro no processamento: {obj.error}")

    if obj.status not in ("done", "partial_data"):
        raise HTTPException(status_code=409, detail="Relatório ainda não está pronto")
    
    try:
        result = json.loads(obj.result_json or "{}")
    except Exception:
        result = {}
        
    return CompetitionReportV2Out(report_id=obj.public_id, status=obj.status, result=result)

def _normalize_nucleus(nucleus: dict) -> dict:
    def norm(v):
        if v is None:
            return "não informado"
        if isinstance(v, str) and not v.strip():
            return "não informado"
        if isinstance(v, list) and len(v) == 0:
            return "não informado"
        return v

    out = {}
    for k, v in (nucleus or {}).items():
        if isinstance(v, dict):
            out[k] = {kk: norm(vv) for kk, vv in v.items()}
        else:
            out[k] = norm(v)
    return out


@app.get("/api/authority-agents/history", response_model=AuthorityAgentHistoryOut)
def authority_agents_history(client_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    items = session.exec(
        select(AuthorityAgentRun)
        .where(AuthorityAgentRun.user_id == current_user.id) # Ignora o client_id do frontend, foca no utilizador real
        .order_by(AuthorityAgentRun.created_at.desc())
        .limit(50)
    ).all()

    return {
        "items": [
            {
                "id": r.id,
                "agent_key": r.agent_key,
                "output_text": r.output_text,
                "created_at": r.created_at.isoformat(),
            }
            for r in items
        ]
    }

@app.get("/api/authority-agents/run/{run_id}", response_model=AuthorityAgentRunOut)
def authority_agents_get_run(run_id: int, client_id: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    run = session.get(AuthorityAgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Execução não encontrada.")
    if run.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Acesso negado para esta execução.")
    return {
        "id": run.id,
        "agent_key": run.agent_key,
        "output_text": run.output_text,
        "created_at": run.created_at.isoformat(),
    }

@app.post("/api/authority-agents/run", response_model=AuthorityAgentRunOut)
def authority_agents_run(payload: AuthorityAgentRunIn, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    
    # NOVO: Verificação de créditos antes de rodar a IA
    if current_user.credits < 5:
        raise HTTPException(status_code=403, detail="Créditos insuficientes. Precisas de 5 créditos para executar o agente de autoridade.")
        
    nucleus = _normalize_nucleus(payload.nucleus)

    global_robot = session.exec(select(Robot).where(Robot.public_id == "business-core-global")).first()
    if global_robot:
        core = session.exec(select(BusinessCore).where(BusinessCore.robot_id == global_robot.id)).first()
        if core and getattr(core, "knowledge_text", None):
            nucleus["conhecimento_anexado"] = core.knowledge_text

    try:
        output = run_authority_agent(payload.agent_key, nucleus)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # NOVO: Descontar os créditos pelo sucesso da execução
    current_user.credits -= 5
    session.add(current_user)

    run = AuthorityAgentRun(
        user_id=current_user.id,
        client_id=payload.client_id, # Mantém por retrocompatibilidade com o request antigo
        agent_key=payload.agent_key,
        nucleus_json=json.dumps(nucleus, ensure_ascii=False),
        output_text=output,
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    return {
        "id": run.id,
        "agent_key": run.agent_key,
        "output_text": run.output_text,
        "created_at": run.created_at.isoformat(),
    }

@app.post("/api/robots/{public_id}/authority-agents/run", response_model=AuthorityAgentRunOut)
def authority_agents_run_compat(public_id: str, payload: AuthorityAgentRunIn, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return authority_agents_run(payload, session, current_user)

@app.get("/api/robots/{public_id}/authority-agents/cooldown")
def authority_agents_cooldown(public_id: str, agent_key: str, client_id: str = "", session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    # Deixamos a devolver sempre 0 para garantir retrocompatibilidade e não quebrar frontends desatualizados.
    return {"cooldown_seconds": 0}

@app.post("/api/robots/{public_id}/upload-knowledge", response_model=RobotDetail)
async def upload_robot_knowledge(
    public_id: str, 
    file: UploadFile = File(...), 
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    robot = _get_robot_or_404(public_id, session, current_user)
    
    text = await extract_text_from_file(file)
    if not text:
        raise HTTPException(status_code=400, detail="Ficheiro vazio ou sem texto legível.")
        
    separator = f"\n\n=== CONTEÚDO DO FICHEIRO: {file.filename} ===\n"
    robot.system_instructions += f"{separator}{text}"
    
    try:
        files_list = json.loads(robot.knowledge_files_json or "[]")
    except:
        files_list = []
    
    files_list.append({"filename": file.filename, "uploaded_at": datetime.utcnow().isoformat()})
    robot.knowledge_files_json = json.dumps(files_list, ensure_ascii=False)
    
    session.add(robot)
    session.commit()
    session.refresh(robot)
    
    return RobotDetail(
        public_id=robot.public_id,
        title=robot.title,
        description=robot.description or "",
        avatar_data=robot.avatar_data,
        system_instructions=robot.system_instructions,
        created_at=robot.created_at.isoformat(),
        knowledge_files_json=robot.knowledge_files_json
    )

@app.delete("/api/robots/{public_id}/knowledge-files/{filename}")
def delete_robot_file(public_id: str, filename: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    """Apaga um ficheiro de conhecimento do robô."""
    robot = _get_robot_or_404(public_id, session, current_user)
    try:
        files_list = json.loads(robot.knowledge_files_json or "[]")
    except:
        files_list = []
        
    new_list = [f for f in files_list if f.get("filename") != filename]
    robot.knowledge_files_json = json.dumps(new_list, ensure_ascii=False)
    
    # Remove do texto mestre
    if robot.system_instructions:
        pattern = rf"\n\n=== CONTEÚDO DO FICHEIRO: {re.escape(filename)} ===\n.*?(?=\n\n=== CONTEÚDO DO FICHEIRO:|$)"
        robot.system_instructions = re.sub(pattern, "", robot.system_instructions, flags=re.DOTALL)
        
    session.add(robot)
    session.commit()
    return {"ok": True}

@app.post("/api/robots/{public_id}/business-core/upload-knowledge")
async def upload_business_core_knowledge(
    public_id: str, 
    file: UploadFile = File(...), 
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    robot = _get_robot_or_404(public_id, session, current_user)
    core = session.exec(select(BusinessCore).where(BusinessCore.robot_id == robot.id)).first()
    
    if not core:
        core = BusinessCore(robot_id=robot.id)
        session.add(core)
        
    text = await extract_text_from_file(file)
    if not text:
        raise HTTPException(status_code=400, detail="Ficheiro vazio ou sem texto.")
        
    current_knowledge = getattr(core, 'knowledge_text', '') or ""
    separator = f"\n\n=== MATERIAIS DE APOIO: {file.filename} ===\n"
    core.knowledge_text = f"{current_knowledge}{separator}{text}"
    
    try:
        files_list = json.loads(core.knowledge_files_json or "[]")
    except:
        files_list = []
        
    files_list.append({"filename": file.filename, "uploaded_at": datetime.utcnow().isoformat()})
    core.knowledge_files_json = json.dumps(files_list, ensure_ascii=False)
    
    core.updated_at = datetime.utcnow()
    session.add(core)
    session.commit()
    session.refresh(core)
    
    return core

@app.delete("/api/robots/{public_id}/business-core/files/{filename}")
def delete_business_core_file(public_id: str, filename: str, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    """Apaga um ficheiro de conhecimento do Núcleo."""
    robot = _get_robot_or_404(public_id, session, current_user)
    core = session.exec(select(BusinessCore).where(BusinessCore.robot_id == robot.id)).first()
    
    if not core:
        raise HTTPException(status_code=404, detail="Núcleo não encontrado")
        
    try:
        files_list = json.loads(core.knowledge_files_json or "[]")
    except:
        files_list = []
        
    new_list = [f for f in files_list if f.get("filename") != filename]
    core.knowledge_files_json = json.dumps(new_list, ensure_ascii=False)
    
    if getattr(core, 'knowledge_text', None):
        pattern = rf"\n\n=== MATERIAIS DE APOIO: {re.escape(filename)} ===\n.*?(?=\n\n=== MATERIAIS DE APOIO:|$)"
        core.knowledge_text = re.sub(pattern, "", core.knowledge_text, flags=re.DOTALL)
        
    session.add(core)
    session.commit()
    return {"ok": True}
@app.post("/api/authority-agents/suggest-themes")
def authority_agents_suggest_themes(payload: SuggestThemesRequest, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    from .ai import suggest_themes_for_task
    try:
        themes = suggest_themes_for_task(payload.agent_key, payload.nucleus, payload.task)
        return {"themes": themes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))