from __future__ import annotations
import json
import uuid
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from .db import init_db, get_session
from .models import Robot, ChatMessage, CompetitionAnalysis, AuthorityEdit, AuthorityAgentRun
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
    CompetitionFindOut, # Adicionado import faltante
)
from .ai import build_robot_from_briefing, chat_with_robot, transcribe_audio, find_competitors, build_competition_result, authority_assistant, run_authority_agent

app = FastAPI(title="Authority Robot Panel API")

# FERA MODE: CORS Permissivo para dev. Em prod, restrinja.
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

# --- ROBOT HELPERS ---
def _get_robot_or_404(public_id: str, session: Session) -> Robot:
    robot = session.exec(select(Robot).where(Robot.public_id == public_id)).first()
    if not robot:
        raise HTTPException(status_code=404, detail="Assistente não encontrado")
    return robot

# --- ROBOT ROUTES ---
@app.get("/api/robots", response_model=list[RobotOut])
def list_robots(session: Session = Depends(get_session)):
    robots = session.exec(select(Robot).order_by(Robot.created_at.desc())).all()
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
def get_robot(public_id: str, session: Session = Depends(get_session)):
    robot = _get_robot_or_404(public_id, session)
    return RobotDetail(
        public_id=robot.public_id,
        title=robot.title,
        description=robot.description or "",
        avatar_data=robot.avatar_data,
        system_instructions=robot.system_instructions,
        created_at=robot.created_at.isoformat(),
    )

@app.delete("/api/robots/{public_id}")
def delete_robot(public_id: str, session: Session = Depends(get_session)):
    robot = _get_robot_or_404(public_id, session)
    msgs = session.exec(select(ChatMessage).where(ChatMessage.robot_id == robot.id)).all()
    for m in msgs:
        session.delete(m)
    session.delete(robot)
    session.commit()
    return {"ok": True}

@app.post("/api/robots", response_model=RobotOut)
def create_robot(brief: BriefingIn, session: Session = Depends(get_session)):
    try:
        built = build_robot_from_briefing(brief.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    robot = Robot(
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
def update_robot(public_id: str, body: RobotUpdateIn, session: Session = Depends(get_session)):
    robot = _get_robot_or_404(public_id, session)
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
    )

# --- MESSAGES ROUTES ---
@app.get("/api/robots/{public_id}/messages", response_model=list[ChatMessageOut])
def list_messages(public_id: str, session: Session = Depends(get_session)):
    robot = _get_robot_or_404(public_id, session)
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
def authority_assistant_route(public_id: str, body: AuthorityAssistantIn, session: Session = Depends(get_session)):
    robot = _get_robot_or_404(public_id, session)

    # Fetch recent authority edits to avoid repeating the same updates
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
    before_hash = __import__("hashlib").sha256(before_instructions.encode("utf-8")).hexdigest()

    result = authority_assistant(
        robot_system_instructions=before_instructions,
        user_message=body.message,
        history=body.history or [],
        authority_edits_history=edits_history,
    )

    if result.get("apply_change") and result.get("updated_system_instructions"):
        updated = str(result["updated_system_instructions"])
        after_hash = __import__("hashlib").sha256(updated.encode("utf-8")).hexdigest()

        # Dedupe: if we already have this exact after state, don't store again
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
                # short summary for neuroplasticity
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

    # Always return a valid response (FastAPI validates against response_model)
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
def list_authority_edits(public_id: str, session: Session = Depends(get_session)):
    robot = _get_robot_or_404(public_id, session)
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
def clear_messages(public_id: str, session: Session = Depends(get_session)):
    robot = _get_robot_or_404(public_id, session)
    msgs = session.exec(select(ChatMessage).where(ChatMessage.robot_id == robot.id)).all()
    for m in msgs:
        session.delete(m)
    session.commit()
    return {"ok": True}

@app.patch("/api/robots/{public_id}/messages/{message_id}", response_model=ChatMessageOut)
def update_message(public_id: str, message_id: int, body: MessageUpdateIn, session: Session = Depends(get_session)):
    robot = _get_robot_or_404(public_id, session)
    msg = session.exec(
        select(ChatMessage).where(ChatMessage.id == message_id, ChatMessage.robot_id == robot.id)
    ).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Mensagem não encontrada")
    if msg.role != "user":
        raise HTTPException(status_code=400, detail="Só é permitido editar mensagens do usuário")

    msg.content = body.content
    session.add(msg)
    session.commit()
    session.refresh(msg)
    return ChatMessageOut(id=msg.id, role=msg.role, content=msg.content, created_at=msg.created_at.isoformat())

@app.post("/api/robots/{public_id}/audio", response_model=ChatMessageOut)
async def chat_audio(public_id: str, file: UploadFile = File(...), session: Session = Depends(get_session)):
    robot = _get_robot_or_404(public_id, session)
    audio_bytes = await file.read()
    try:
        text = transcribe_audio(audio_bytes, filename=file.filename or "audio.webm")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao transcrever áudio: {e}")

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
def chat(public_id: str, body: ChatIn, session: Session = Depends(get_session)):
    robot = _get_robot_or_404(public_id, session)
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
    from .db import get_session as _get_s
    gen = _get_s()
    session = next(gen)
    obj = None
    try:
        obj = session.exec(select(CompetitionAnalysis).where(CompetitionAnalysis.public_id == public_id)).first()
        if not obj:
            return
        
        _update_analysis(session, obj, status="running", stage="Coletando sinais públicos", progress=0.15)
        
        # Parse inputs
        instagrams = json.loads(obj.instagrams_json or "[]")
        sites = json.loads(obj.sites_json or "[]")
        company = json.loads(obj.company_json or "{}") if obj.company_json else {}

        competitors = []
        for s in sites:
            competitors.append({"name": _domain(s), "website_url": s})
        
        # Simulação de progresso real
        _update_analysis(session, obj, stage="Analisando presença digital", progress=0.40)
        
        # ... lógica de instagram aqui ...

        _update_analysis(session, obj, stage="Consolidando inteligência", progress=0.75)

        result = build_competition_result(company=company, competitors=competitors[:3])

        _update_analysis(session, obj, status="done", stage="Concluído", progress=1.0, result_json=json.dumps(result, ensure_ascii=False))

    except Exception as e:
        if obj:
            _update_analysis(session, obj, status="error", stage="Erro no processamento", progress=1.0, error=str(e))
        print(f"Job Error: {e}")
    finally:
        session.close()

# --- COMPETITION ROUTES (V2 ONLY) ---

@app.post("/api/competition/find-competitors", response_model=CompetitionFindOut)
def competition_find_competitors_v2(payload: CompetitionFindRequest):
    briefing = payload.briefing.model_dump()
    mapped = {
        "company_name": briefing.get("nome_empresa"),
        "niche": briefing.get("segmento"),
        "region": briefing.get("cidade_estado"),
        "services": briefing.get("servicos"),
        "audience": briefing.get("publico_alvo"),
        "offer": briefing.get("servicos"),
    }
    # Aqui assume-se que find_competitors retorna estrutura compatível com CompetitionFindOut
    data = find_competitors(mapped)
    return data

@app.post("/api/competition/analyze", response_model=CompetitionJobV2Out)
def competition_analyze_v2(payload: CompetitionAnalyzeRequest, bg: BackgroundTasks, session: Session = Depends(get_session)):
    public_id = str(uuid.uuid4())
    instas = payload.instagrams or []
    sites = payload.sites or []
    briefing = payload.briefing.model_dump() if payload.briefing else None

    obj = CompetitionAnalysis(
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
def competition_job_v2(job_id: str, session: Session = Depends(get_session)):
    obj = session.exec(select(CompetitionAnalysis).where(CompetitionAnalysis.public_id == job_id)).first()
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
def competition_report_v2(report_id: str, session: Session = Depends(get_session)):
    obj = session.exec(select(CompetitionAnalysis).where(CompetitionAnalysis.public_id == report_id)).first()
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
    # Preenche campos vazios com "não informado"
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


def _cooldown_check(session: Session, client_id: str, agent_key: str, cooldown_seconds: int = 3600):
    last = session.exec(
        select(AuthorityAgentRun)
        .where(AuthorityAgentRun.client_id == client_id)
        .where(AuthorityAgentRun.agent_key == agent_key)
        .order_by(AuthorityAgentRun.created_at.desc())
    ).first()
    if not last:
        return

    elapsed = (datetime.utcnow() - last.created_at.replace(tzinfo=None)).total_seconds()
    if elapsed < cooldown_seconds:
        retry_after = int(cooldown_seconds - elapsed)
        raise HTTPException(
            status_code=429,
            detail={"message": "Cooldown ativo para este agente.", "retry_after_seconds": retry_after},
            headers={"Retry-After": str(retry_after)},
        )


@app.get("/api/authority-agents/history", response_model=AuthorityAgentHistoryOut)
def authority_agents_history(client_id: str, session: Session = Depends(get_session)):
    items = session.exec(
        select(AuthorityAgentRun)
        .where(AuthorityAgentRun.client_id == client_id)
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
def authority_agents_get_run(run_id: int, client_id: str, session: Session = Depends(get_session)):
    run = session.get(AuthorityAgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Execução não encontrada.")
    if run.client_id != client_id:
        # evita vazamento de dados entre navegadores/usuários
        raise HTTPException(status_code=403, detail="Acesso negado para esta execução.")
    return {
        "id": run.id,
        "agent_key": run.agent_key,
        "output_text": run.output_text,
        "created_at": run.created_at.isoformat(),
    }

@app.post("/api/authority-agents/run", response_model=AuthorityAgentRunOut)
def authority_agents_run(payload: AuthorityAgentRunIn, session: Session = Depends(get_session)):
    nucleus = _normalize_nucleus(payload.nucleus)
    _cooldown_check(session, payload.client_id, payload.agent_key)

    try:
        output = run_authority_agent(payload.agent_key, nucleus)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    run = AuthorityAgentRun(
        client_id=payload.client_id,
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


# Backward-compat: se o front antigo ainda bater em /api/robots/{public_id}/authority-agents/run
@app.post("/api/robots/{public_id}/authority-agents/run", response_model=AuthorityAgentRunOut)
def authority_agents_run_compat(public_id: str, payload: AuthorityAgentRunIn, session: Session = Depends(get_session)):
    return authority_agents_run(payload, session)
