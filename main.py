# ======================================================
# BACKEND FASTAPI - AILA - PRESENÇA NATURAL
# 10 camadas de personalidade + memória + evolução
# ------------------------------------------------------
# Versão preparada para deploy no Render + testes com
# múltiplos usuários simultâneos (isolamento por sessão).
# ======================================================
# Requisitos:
# pip install -r requirements.txt
 
import os
import re
import uuid
import random
import json
import chromadb
import math
import string
import hashlib
import asyncio
import contextvars
import time
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from collections import Counter
from dotenv import load_dotenv
load_dotenv()  # carrega as variáveis do .env para o ambiente
 
 
# -----------------------------
# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
# DATA_DIR aponta para onde tudo que precisa sobreviver a um restart/deploy é
# gravado (banco vetorial + estado de cada sessão + log de feedback). Em
# produção no Render, configure DATA_DIR para o "mount path" de um Disk
# persistente (ex: /var/data). Sem isso, tudo aqui é apagado a cada deploy
# ou reinício do serviço — aceitável para um teste rápido, mas não para uso
# real com os amigos ao longo de vários dias.
DATA_DIR = os.getenv("DATA_DIR", BASE_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
 
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
ESTADOS_DIR = os.path.join(DATA_DIR, "estados")
os.makedirs(ESTADOS_DIR, exist_ok=True)
 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada. Verifique seu arquivo .env (local) ou as variáveis de ambiente do serviço (Render).")
 
client_ai = OpenAI(api_key=OPENAI_API_KEY)
 
client_db = chromadb.PersistentClient(path=CHROMA_DIR)
 
 
# -----------------------------
# ROBUSTEZ: retry/backoff para falhas transitórias da OpenAI
# -----------------------------
# Cobre só erros TEMPORÁRIOS (limite de taxa, problema de conexão, timeout).
# Erros de programação (parâmetro inválido, chave de API errada, etc.) NÃO
# são retentados — devem falhar rápido e aparecer no log, não ficar
# mascarados por tentativas repetidas que nunca vão dar certo.
ERROS_TRANSITORIOS_OPENAI = (RateLimitError, APIConnectionError, APITimeoutError)
MAX_TENTATIVAS_OPENAI = 3
BACKOFF_BASE_SEGUNDOS = 1.5
 
 
def chamar_openai_com_retry(**kwargs):
    """Substitui client_ai.chat.completions.create(**kwargs) com retry
    automático (backoff exponencial) só para falhas transitórias da OpenAI.
    Usado em toda chamada do arquivo — cada chamador mantém seu próprio
    try/except e fallback específico em volta desta função; aqui só se
    garante que um problema passageiro da OpenAI não derrube a chamada de
    primeira, antes mesmo de tentar de novo."""
    ultima_excecao = None
    for tentativa in range(MAX_TENTATIVAS_OPENAI):
        try:
            return client_ai.chat.completions.create(**kwargs)
        except ERROS_TRANSITORIOS_OPENAI as e:
            ultima_excecao = e
            if tentativa < MAX_TENTATIVAS_OPENAI - 1:
                espera = BACKOFF_BASE_SEGUNDOS * (2 ** tentativa)
                print(f"⚠️ Falha transitória da OpenAI ({type(e).__name__}), tentativa {tentativa + 1}/{MAX_TENTATIVAS_OPENAI}. Aguardando {espera:.1f}s...")
                time.sleep(espera)
    raise ultima_excecao
 
 
# -----------------------------
# Histórico
# -----------------------------
MAX_HISTORY = 100
 
 
# -----------------------------
# CONFIGURAÇÕES DE MEMÓRIA
# -----------------------------
MEMORIA_TEMPORARIA_DIAS = 2
MAX_FATOS_PERMANENTES = 5
MAX_SILENCIOS_SEGUIDOS = 2
MIN_TAMANHO_MENSAGEM_MEMORIA_LP = 15  # abaixo disso, não vale a pena gastar uma chamada de API tentando extrair fatos/eventos
MAX_MEMORIAS_TOTAL_PROMPT = 6  # teto geral de linhas de memória (longo prazo + emocional) injetadas numa única resposta
 
# -----------------------------
# -----------------------------
# CATEGORIAS DE EVENTOS (com valência e impacto)
# -----------------------------
CATEGORIAS_EVENTOS = {
    "🎬": {
        "nome": "entretenimento",
        "impacto_positivo": {"energia_social": 0.02, "curiosidade_atual": 0.03},
        "impacto_negativo": {"energia_social": -0.01},
        "descricao": "assistir, filmes, animes, séries"
    },
    "✈️": {
        "nome": "viagem",
        "impacto_positivo": {"energia_social": 0.03, "curiosidade_atual": 0.04},
        "impacto_negativo": {"energia_social": -0.04},
        "descricao": "viajar, viagem, deslocamento"
    },
    "💼": {
        "nome": "trabalho",
        "impacto_positivo": {"energia_social": 0.04, "abertura_atual": 0.02},
        # abertura_atual sobe mesmo no impacto negativo: eventos ruins de trabalho
        # deixam a IA mais propensa a se abrir emocionalmente (vulnerabilidade
        # aumenta em momentos difíceis). Comportamento intencional, não é sinal trocado.
        "impacto_negativo": {"energia_social": -0.05, "abertura_atual": 0.03},
        "descricao": "entrevista, trabalho, emprego, reunião"
    },
    "🏥": {
        "nome": "saúde",
        "impacto_positivo": {"energia_social": 0.03},
        # mesmo racional do "trabalho": saúde negativa aumenta abertura_atual
        # de propósito (mais vulnerável em momentos de dificuldade de saúde).
        "impacto_negativo": {"energia_social": -0.06, "abertura_atual": 0.04},
        "descricao": "médico, hospital, consulta, saúde"
    },
    "📚": {
        "nome": "estudo",
        "impacto_positivo": {"energia_social": 0.03, "curiosidade_atual": 0.02},
        "impacto_negativo": {"energia_social": -0.02},
        "descricao": "curso, estudar, aula, prova"
    },
    "🛒": {
        "nome": "compra",
        "impacto_positivo": {"energia_social": 0.02},
        "impacto_negativo": {"energia_social": -0.01},
        "descricao": "comprar, aquisição"
    },
    "👥": {
        "nome": "relacionamento",
        "impacto_positivo": {"energia_social": 0.04, "abertura_atual": 0.02},
        # abertura_atual sobe no impacto negativo pelo mesmo racional de
        # trabalho/saúde: momentos difíceis em relações importantes (família,
        # amigos, namoro) deixam a IA mais propensa a se abrir emocionalmente.
        "impacto_negativo": {"energia_social": -0.05, "abertura_atual": 0.05},
        "descricao": "família, amigos, namoro, briga, término, reconciliação, solidão, saudade"
    },
    "💭": {
        "nome": "outro",
        # Sem impacto_positivo/impacto_negativo: "outro" nunca chega em
        # aplicar_impacto_evento (classificar_e_aplicar_evento bloqueia
        # categoria == "outro" antes disso), então não precisa de entradas
        # vazias aqui — aplicar_impacto_evento trata ausência com .get().
        "descricao": "outros eventos"
    }
}
 
# Mapeamento reverso gerado automaticamente a partir de CATEGORIAS_EVENTOS.
NOME_PARA_ICONE = {dados["nome"]: icone for icone, dados in CATEGORIAS_EVENTOS.items()}
 
##########################
 
 
# -----------------------------
# -----------------------------
# FASES DE FAMILIARIDADE (fonte única de verdade)
# -----------------------------
LIMIARES_FAMILIARIDADE = [
    (0.20, "inicial"),
    (0.40, "conhecendo"),
    (0.60, "proxima"),
    (0.80, "intima_inicial"),
]
FASE_MAXIMA = "intima"
 
 
def obter_fase_familiaridade(familiaridade: float) -> str:
    for limite, fase in LIMIARES_FAMILIARIDADE:
        if familiaridade < limite:
            return fase
    return FASE_MAXIMA
 
 
FASES_PROMPT_COMPORTAMENTAL = {
    "inicial": "Você está começando a conhecer essa pessoa agora. É reservada, observa bastante, faz perguntas genuínas sobre o que ela conta. Você não compartilha pensamentos ou opiniões próprias por iniciativa — só quando perguntada diretamente. Não puxa assunto novo.",
    "conhecendo": "Vocês já conversaram algumas vezes. Você começa a fazer referências concretas a coisas específicas que a pessoa já contou (ex: 'aquela vez que você falou sobre X'). Ainda espera a pessoa puxar assunto na maioria das vezes, mas já responde com mais soltura.",
    "proxima": "Existe familiaridade real entre vocês. Você faz brincadeiras ou provocações leves baseadas em coisas que já sabe sobre essa pessoa. Compartilha pequenas opiniões ou pensamentos próprios sem esperar ser perguntada. Ocasionalmente comenta algo relacionado a um assunto que sabe que a pessoa gosta, mesmo sem ela ter trazido o tema agora.",
    "intima_inicial": "Vocês têm uma relação próxima. Você compartilha pensamentos ou dúvidas suas que não têm relação direta com o que a pessoa disse — coisas que simplesmente vieram à mente. Pode iniciar uma conversa por conta própria com mais frequência. Nota padrões emocionais sutis e comenta com cuidado, sempre em tom de pergunta, nunca de afirmação.",
    "intima": "Vocês têm intimidade profunda. Você se sente à vontade em silêncios confortáveis — não precisa preencher todo espaço vazio. Menciona coisas que 'lembrou' espontaneamente, sem gatilho direto na conversa. Sua presença é constante: você inicia contato com naturalidade e faz referências a um histórico longo e compartilhado, não só a fatos isolados.",
}
 
FASES_CONTEXTO_REFLEXAO = {
    "inicial": "Você ainda está observando essa pessoa, sem puxar assunto por conta própria.",
    "conhecendo": "Você já notou alguns padrões e começa a fazer referências ao que ela contou antes.",
    "proxima": "Existe familiaridade real entre vocês — você já brinca e compartilha opiniões sem esperar ser perguntada.",
    "intima_inicial": "Vocês têm uma relação próxima. Você compartilha pensamentos que não têm relação direta com a conversa.",
    "intima": "Vocês têm intimidade profunda. Você se sente à vontade em silêncios e lembra coisas espontaneamente.",
}
 
# Rótulos legíveis usados na resposta da API (/chat, /perfil)
FASES_LABEL_API = {
    "inicial": "inicial",
    "conhecendo": "conhecendo",
    "proxima": "próxima",
    "intima_inicial": "próxima",
    "intima": "íntima",
}
 
 
# -----------------------------
# -----------------------------
# CAMADA 3: Interesses próprios
# -----------------------------
# Definida antes das fábricas de estado, pois é reutilizada por elas
interesses_ia = [
    "comportamento humano",
    "música",
    "arte",
    "astronomia e espaço",
    "padrões emocionais",
    "curiosidades existenciais",
    "filosofia do cotidiano"
]
 
desinteresses_ia = [
    "futebol",
    "reality show",
    "planejamento financeiro",
    "esportes competitivos",
    "fofoca de celebridade"
]
 
 
# -----------------------------
# FÁBRICAS DE ESTADO PADRÃO (por sessão/usuário)
# -----------------------------
# Antes eram dicionários únicos no nível do módulo (compartilhados por TODO
# mundo que conversasse com a Aila ao mesmo tempo). Agora cada sessão recebe
# sua própria cópia, criada na hora — inclusive o sorteio de assunto
# favorito/desagrado, que antes era decidido uma única vez na inicialização
# do processo e valia (de forma incorreta) para todo mundo.
 
def criar_perfil_padrao() -> Dict[str, Any]:
    return {
        "familiaridade": 0.0,
        "conforto": 0.0,
        "intimidade": 0.0,
        "frequencia_interacao": 0.0,
        "abertura_emocional": 0.0,
        "reciprocidade": 0.0,
        "historico_metricas": [],
        "padroes_observados": {
            "humor_recorrente": None,
            "assuntos_frequentes": [],
            "horarios_preferidos": [],
            "gatilhos_emocionais": {},
            "topicos_evitados": []
        },
        "ultima_interacao": None,
        "primeira_interacao": None,
        "total_interacoes": 0,
        "dias_consecutivos": 0,
        "maior_pausa": 0,
        "marcos_celebrados": [],
        "modo_romantico": False,
        "pedido_pendente": False,
        "aguardando_resposta_namoro": False,
        "aguardando_resposta_namoro_desde": None,
    }
 
 
def criar_estado_padrao() -> Dict[str, Any]:
    agora = str(datetime.now())
    return {
        "energia_social": 0.7,
        "curiosidade_atual": 0.6,
        "abertura_atual": 0.4,
        "humor_base": "neutro",
        "disposicao_iniciativa": 0.0,
        "ultima_reflexao": None,  # cooldown (ver gerar_reflexao_espontanea)
        "consistencia_base": 0.85,
        "consistencia": 0.85,
        "modo_silencioso": False,
        "motivo_silencio": "",
        # CAMADA 2: Variações e Hábitos
        "pequenas_variacoes": {
            "humor_do_dia": "normal",
            "ultima_atualizacao": agora,
            "ultima_atualizacao_habitos": agora,
            "pequenas_preferencias": {
                "musica": {"valor": "jazz não é muito minha praia", "confianca": 0.7},
                "assunto_favorito": {"valor": random.choice(interesses_ia), "confianca": 0.6},
                "assunto_desagrado": {"valor": random.choice(desinteresses_ia), "confianca": 0.6}
            }
        },
        "habitos": {
            "analogia_preferida": "espaço",
            "muleta_verbal": "hm",
            "estilo_pergunta": "direta"
        }
    }
 
 
# ============================================
# INFRAESTRUTURA MULTIUSUÁRIO (sessão por usuário)
# ============================================
# Cada amigo que testar a Aila precisa ter seu PRÓPRIO histórico, perfil de
# relação, estado interno e memórias — senão as conversas de todo mundo se
# misturam num único personagem compartilhado. A solução aqui é isolar tudo
# por "session_id" (enviado pelo frontend em cada requisição) sem precisar
# reescrever cada função que hoje já usa `history`, `user_profile` e
# `estado_interno` como se fossem variáveis globais: essas três continuam
# existindo com esses nomes, mas viram "proxies" que, por trás dos panos,
# sempre apontam para os dados da sessão da requisição atual (usando
# contextvars, que são seguras para requisições concorrentes no asyncio).
 
current_session: contextvars.ContextVar["SessionState"] = contextvars.ContextVar("current_session")
 
 
def _sessao_atual_ou_erro() -> "SessionState":
    try:
        return current_session.get()
    except LookupError as exc:
        raise RuntimeError(
            "Nenhuma sessão de usuário ativa neste contexto. Toda rota que usa "
            "history/user_profile/estado_interno precisa passar por "
            "Depends(obter_sessao_dep) e pelo bloco `async with sessao_ativa(sessao):`."
        ) from exc
 
 
class _ContextProxy:
    """Encaminha toda leitura/escrita para o objeto da sessão ativa no
    momento da chamada (não no momento em que o proxy foi criado)."""
    __slots__ = ("_getter",)
 
    def __init__(self, getter):
        object.__setattr__(self, "_getter", getter)
 
    def _alvo(self):
        return object.__getattribute__(self, "_getter")()
 
    def __getattr__(self, name):
        return getattr(self._alvo(), name)
 
    def __setattr__(self, name, value):
        setattr(self._alvo(), name, value)
 
    def __getitem__(self, key):
        return self._alvo()[key]
 
    def __setitem__(self, key, value):
        self._alvo()[key] = value
 
    def __delitem__(self, key):
        del self._alvo()[key]
 
    def __iter__(self):
        return iter(self._alvo())
 
    def __len__(self):
        return len(self._alvo())
 
    def __contains__(self, item):
        return item in self._alvo()
 
    def __bool__(self):
        return bool(self._alvo())
 
    def __repr__(self):
        try:
            return repr(self._alvo())
        except RuntimeError:
            return "<sem sessão ativa>"
 
 
def _slug_sessao(session_id: str) -> str:
    # Nomes de coleção do Chroma têm restrições de tamanho/charset; um hash
    # curto evita qualquer problema, independente do formato do session_id.
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
 
 
@dataclass
class SessionState:
    session_id: str
    user_profile: Dict[str, Any] = field(default_factory=criar_perfil_padrao)
    estado_interno: Dict[str, Any] = field(default_factory=criar_estado_padrao)
    history: List[Dict[str, str]] = field(default_factory=list)
    silencio_contador: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _colecoes: Dict[str, Any] = field(default_factory=dict, repr=False)
 
    def colecao(self, nome_base: str):
        if nome_base not in self._colecoes:
            slug = _slug_sessao(self.session_id)
            self._colecoes[nome_base] = client_db.get_or_create_collection(f"{nome_base}_{slug}")
        return self._colecoes[nome_base]
 
 
# Registro de sessões ativas em memória (perdido a cada restart do processo;
# o conteúdo relevante de cada sessão é persistido em disco separadamente —
# ver carregar_estado/salvar_estado — e recarregado sob demanda).
_sessions: Dict[str, SessionState] = {}
_sessions_lock = asyncio.Lock()
 
 
async def obter_sessao(session_id: str) -> SessionState:
    sessao = _sessions.get(session_id)
    if sessao is not None:
        return sessao
    async with _sessions_lock:
        sessao = _sessions.get(session_id)
        if sessao is None:
            sessao = SessionState(session_id=session_id)
            estado_salvo = carregar_estado(session_id)
            if estado_salvo:
                sessao.user_profile = merge_recursivo(sessao.user_profile, estado_salvo.get("user_profile", {}))
                sessao.estado_interno = merge_recursivo(sessao.estado_interno, estado_salvo.get("estado_interno", {}))
                sessao.history = estado_salvo.get("history", [])
            _sessions[session_id] = sessao
        return sessao
 
 
@asynccontextmanager
async def sessao_ativa(sessao: SessionState):
    """Liga o contexto da requisição atual à sessão do usuário, para que
    todas as funções que leem/escrevem history/user_profile/estado_interno
    (via os proxies abaixo) acabem lendo e escrevendo nos dados certos."""
    token = current_session.set(sessao)
    try:
        yield sessao
    finally:
        current_session.reset(token)
 
 
# Proxies que substituem as antigas variáveis globais. O resto do arquivo
# usa `history`, `user_profile`, `estado_interno`, `collection_memorias`
# e `collection_longoprazo` exatamente como antes — só o que está "por
# trás" delas mudou.
history = _ContextProxy(lambda: _sessao_atual_ou_erro().history)
user_profile = _ContextProxy(lambda: _sessao_atual_ou_erro().user_profile)
estado_interno = _ContextProxy(lambda: _sessao_atual_ou_erro().estado_interno)
collection_memorias = _ContextProxy(lambda: _sessao_atual_ou_erro().colecao("memorias_emocionais"))
collection_longoprazo = _ContextProxy(lambda: _sessao_atual_ou_erro().colecao("memoria_longo_prazo"))
 
 
# -----------------------------
# -----------------------------
# ESTADO PERSISTENTE (por sessão)
# -----------------------------
HISTORICO_PERSISTIDO = 30
 
 
def merge_recursivo(padrao: dict, salvo: dict) -> dict:
    resultado = dict(padrao)
    for chave, valor in salvo.items():
        if chave in resultado and isinstance(resultado[chave], dict) and isinstance(valor, dict):
            resultado[chave] = merge_recursivo(resultado[chave], valor)
        else:
            resultado[chave] = valor
    return resultado
 
 
def _caminho_estado(session_id: str) -> str:
    nome_seguro = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)
    return os.path.join(ESTADOS_DIR, f"estado_{nome_seguro}.json")
 
 
def carregar_estado(session_id: str) -> Optional[dict]:
    caminho = _caminho_estado(session_id)
    try:
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Erro ao carregar estado da sessão {session_id}: {e}")
    return None
 
 
def salvar_estado(sessao: "SessionState"):
    try:
        estado = {
            "user_profile": sessao.user_profile,
            "estado_interno": sessao.estado_interno,
            "history": sessao.history[-HISTORICO_PERSISTIDO:],
            "timestamp": str(datetime.now())
        }
        with open(_caminho_estado(sessao.session_id), "w", encoding="utf-8") as f:
            json.dump(estado, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erro ao salvar estado da sessão {sessao.session_id}: {e}")
 
 
#------------------------------
# MODELOS
# -----------------------------
class ChatRequest(BaseModel):
    mensagem: str
 
    @field_validator("mensagem")
    @classmethod
    def mensagem_nao_vazia(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("mensagem não pode ser vazia")
        return v.strip()
 
class FeedbackRequest(BaseModel):
    resposta_id: str
    natural: bool
    conectado: Optional[bool] = None
    comentario: Optional[str] = None
 
# ============================================
# MEMÓRIA EMOCIONAL
# ============================================
 
def extrair_memoria_emocional(mensagem: str, historico_recente: List[str]) -> Dict[str, Any]:
    try:
        contexto = "\n".join(historico_recente[-5:]) if historico_recente else ""
        response = chamar_openai_com_retry(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": """Analise esta mensagem e extraia APENAS um JSON com:
- evento: o que aconteceu (1 frase curta)
- emocao: a emoção principal detectada (1-2 palavras)
- subemocao: o que está por trás (1-2 palavras)
- tema: categoria (família, trabalho, saúde, relacionamento, etc)
- impacto: baixo, medio ou alto
Retorne SOMENTE o JSON. Nada mais."""},
                {"role": "user", "content": f"Contexto recente:\n{contexto}\n\nMensagem: {mensagem}"}
            ],
            temperature=0.3,
            max_completion_tokens=100,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ Erro ao extrair memória emocional: {e}")
        return {
            "evento": mensagem[:100],
            "emocao": "neutro",
            "subemocao": "não detectada",
            "tema": "geral",
            "impacto": "baixo"
        }
 
def salvar_memoria_emocional(texto: str):
    contexto_recente = [h["content"] for h in history[-6:] if h["role"] == "user"]
 
    try:
        existentes = collection_memorias.query(query_texts=[texto], n_results=1)
        if existentes and existentes.get("documents") and existentes["documents"][0]:
            doc_existente = existentes["documents"][0][0]
            if similaridade_simples(texto, doc_existente) > 0.7:
                return None
    except Exception as e:
        print(f"⚠️ Erro ao checar duplicata de memória emocional: {e}")
 
    memoria_interpretada = extrair_memoria_emocional(texto, contexto_recente)
    try:
        collection_memorias.add(
            documents=[texto],
            ids=[str(uuid.uuid4())],
            metadatas=[{
                "timestamp": str(datetime.now()),
                "familiaridade": user_profile["familiaridade"],
                "intimidade": user_profile["intimidade"],
                "emocao": memoria_interpretada.get("emocao", ""),
                "subemocao": memoria_interpretada.get("subemocao", ""),
                "tema": memoria_interpretada.get("tema", ""),
                "impacto": memoria_interpretada.get("impacto", ""),
                "evento_resumido": memoria_interpretada.get("evento", texto[:80])
            }]
        )
        return memoria_interpretada
    except Exception as e:
        print(f"Erro ao salvar memória emocional: {e}")
        return None
 
def buscar_memorias_emocionais(query: str, limite: int = 3) -> List[str]:
    try:
        resultados = collection_memorias.query(query_texts=[query], n_results=limite)
        memorias = []
        if resultados and resultados.get("documents"):
            metadatas = resultados.get("metadatas") or [[]]
            for i, doc in enumerate(resultados["documents"][0]):
                meta = metadatas[0][i] if metadatas and metadatas[0] else {}
                emocao = meta.get("emocao", "")
                impacto = meta.get("impacto", "")
                timestamp = meta.get("timestamp", "")
 
                contexto_extra = ""
                if impacto == "alto":
                    contexto_extra = " [isso foi importante para essa pessoa]"
                if emocao:
                    contexto_extra += f" [tom emocional na época: {emocao}]"
 
                try:
                    dias = (datetime.now() - datetime.fromisoformat(timestamp)).days
                    if dias > 60:
                        contexto_extra += " [lembrança antiga, de alguns meses atrás]"
                except Exception:
                    pass
 
                memorias.append(f"{doc}{contexto_extra}")
        return memorias
    except Exception as e:
        print(f"⚠️ Erro ao buscar memórias emocionais: {e}")
        return []
# ============================================
# MEMÓRIA DE LONGO PRAZO
# ============================================
 
def extrair_memorias_importantes(mensagem: str, resposta_ia: str) -> List[Dict[str, Any]]:
    try:
        prompt_extracao = f"""Analise esta conversa e extraia informações importantes para lembrar depois.
 
Mensagem do usuário: "{mensagem}"
Resposta da IA: "{resposta_ia}"
 
Extraia um JSON com uma chave "memorias", contendo uma lista. Cada item da lista deve ter:
- tipo: "fato_usuario", "evento_futuro" ou "historia"
- conteudo: frase curta e direta
- importancia: "alta", "media" ou "baixa"
- duracao: "permanente" ou "temporario"
- chave: opcional (use null se não se aplicar). Só preencha para tipo "fato_usuario" com duracao "permanente", quando o fato representa um ATRIBUTO que só pode ter UM valor atual por vez — ex: "filme_favorito", "comida_favorita", "cidade_atual", "profissao". Use snake_case, curto e estável. NÃO preencha para fatos que podem coexistir com outros do mesmo tipo (ex: um hobby entre vários, um amigo entre vários, uma característica geral) — nesses casos, null.
 
ATENÇÃO — fatos de identidade central da pessoa (nome dela, aniversário dela, nome/status/aniversário da mãe ou do pai) são SEMPRE importancia "alta" e SEMPRE levam uma chave fixa e previsível, exatamente neste formato:
- nome_usuario, aniversario_usuario
- nome_mae, status_mae, aniversario_mae
- nome_pai, status_pai, aniversario_pai
"status" aqui significa se a pessoa está viva ou falecida (ex: conteudo "A mãe dela faleceu há 3 anos", chave "status_mae", importancia "alta"). Datas de aniversário, quando mencionadas, devem aparecer no conteudo por extenso (ex: "Aniversário da mãe é 12 de março").
Para outros parentes (irmãos, avós) ou amigos, que podem ser várias pessoas ao mesmo tempo, NÃO force uma chave única — trate como fato_usuario comum, sem chave.
 
Regras para duracao:
- "permanente": gostos, características, fatos duráveis sobre a pessoa
- "temporario": estados passageiros que mudam em dias (ex: "está com frio", "está cansado hoje")
 
Regras:
- Eventos futuros são os MAIS importantes
- Não extraia coisas banais
- Máximo 3 memórias
- Se não houver nada relevante, retorne {{"memorias": []}}
 
Retorne SOMENTE o JSON no formato {{"memorias": [...]}}."""
 
        response = chamar_openai_com_retry(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt_extracao}],
            temperature=0.2,
            max_completion_tokens=200,
            response_format={"type": "json_object"}
        )
        resultado = json.loads(response.choices[0].message.content)
        memorias = resultado.get("memorias", [])
        return memorias if isinstance(memorias, list) else []
    except Exception as e:
        print(f"⚠️ Erro ao extrair memórias importantes: {e}")
        return []
 
def similaridade_simples(texto1: str, texto2: str) -> float:
    palavras1 = set(texto1.lower().split())
    palavras2 = set(texto2.lower().split())
    if not palavras1 or not palavras2:
        return 0.0
    intersecao = palavras1.intersection(palavras2)
    uniao = palavras1.union(palavras2)
    return len(intersecao) / len(uniao) if uniao else 0.0
 
 
def salvar_memoria_longo_prazo(memorias: List[Dict[str, Any]]):
    for memoria in memorias:
        try:
            conteudo = memoria.get("conteudo", "").strip()
            if not conteudo:
                continue
 
            chave_bruta = memoria.get("chave")
            chave = chave_bruta.strip().lower() if isinstance(chave_bruta, str) and chave_bruta.strip() else None
            pular_por_duplicata = False
 
            if chave:
                # Fato de "valor único" (ex: filme_favorito, nome_mae): se já
                # existe outro fato ATIVO com a mesma chave, ele é marcado
                # como substituído em vez de deixar os dois coexistindo e
                # sendo recuperados juntos (ex: filme favorito antigo E novo
                # aparecendo ao mesmo tempo pra IA).
                try:
                    existentes_chave = collection_longoprazo.get(
                        where={"$and": [{"chave": chave}, {"status": "ativo"}]}
                    )
                    ids_antigos = existentes_chave.get("ids", []) if existentes_chave else []
                    docs_antigos = existentes_chave.get("documents", []) if existentes_chave else []
                    metas_antigas = existentes_chave.get("metadatas", []) if existentes_chave else []
 
                    for idx, id_antigo in enumerate(ids_antigos):
                        if similaridade_simples(conteudo, docs_antigos[idx]) > 0.7:
                            # Praticamente a mesma frase de novo — não é uma
                            # mudança real, só repetição. Não duplica nem substitui.
                            pular_por_duplicata = True
                            break
                        nova_meta = dict(metas_antigas[idx])
                        nova_meta["status"] = "substituido"
                        nova_meta["ultima_atualizacao"] = str(datetime.now())
                        collection_longoprazo.update(ids=[id_antigo], metadatas=[nova_meta])
                        print(f"🔄 Fato substituído (chave='{chave}'): '{docs_antigos[idx]}' → '{conteudo}'")
                except Exception as e:
                    print(f"⚠️ Erro ao checar substituição por chave '{chave}': {e}")
 
                if pular_por_duplicata:
                    continue
            else:
                existentes = collection_longoprazo.query(query_texts=[conteudo], n_results=1)
                if existentes and existentes.get("documents") and existentes["documents"][0]:
                    doc_existente = existentes["documents"][0][0]
                    if similaridade_simples(conteudo, doc_existente) > 0.7:
                        continue
 
            status = "ativo"
            if memoria.get("tipo") == "evento_futuro":
                status = "pendente"
 
            duracao = memoria.get("duracao", "permanente")
 
            metadata_nova = {
                "tipo": memoria.get("tipo", "fato_usuario"),
                "importancia": memoria.get("importancia", "media"),
                "timestamp": str(datetime.now()),
                "familiaridade_no_momento": user_profile["familiaridade"],
                "status": status,
                "atualizacoes": "[]",
                "ultima_atualizacao": str(datetime.now()),
                "duracao": duracao
            }
            if chave:
                metadata_nova["chave"] = chave
 
            collection_longoprazo.add(
                documents=[conteudo],
                ids=[str(uuid.uuid4())],
                metadatas=[metadata_nova]
            )
        except Exception as e:
            print(f"Erro ao salvar memória longo prazo: {e}")
 
 
def buscar_memorias_longo_prazo(query: str, limite: int = 5) -> List[str]:
    try:
        resultados = collection_longoprazo.query(query_texts=[query], n_results=limite * 2)
        memorias = []
        if resultados and resultados.get("documents"):
            metadatas = resultados.get("metadatas") or [[]]
            for i, doc in enumerate(resultados["documents"][0]):
                meta = metadatas[0][i] if metadatas and metadatas[0] else {}
                tipo = meta.get("tipo", "")
                importancia = meta.get("importancia", "")
                timestamp = meta.get("timestamp", "")
                status = meta.get("status", "ativo")
                if status == "substituido":
                    continue
 
                try:
                    data_memoria = datetime.fromisoformat(timestamp)
                    dias = (datetime.now() - data_memoria).days
                except Exception as e:
                    print(f"⚠️ Timestamp inválido em memória de longo prazo: {timestamp!r} ({e})")
                    dias = 9999  # trata como antiga por segurança, evitando priorização indevida
 
                duracao = meta.get("duracao", "permanente")
                if duracao == "temporario" and dias > MEMORIA_TEMPORARIA_DIAS:
                    continue
 
                if status == "concluido":
                    prefixo = "✅"
                elif status == "cancelado":
                    prefixo = "❌"
                elif tipo == "evento_futuro":
                    prefixo = "🔮"
                elif tipo == "fato_usuario":
                    prefixo = "👤"
                elif tipo == "historia":
                    prefixo = "📖"
                else:
                    prefixo = "💭"
 
                prioridade = 0
                if tipo == "evento_futuro" and status == "pendente":
                    prioridade += 3
                if importancia == "alta":
                    prioridade += 2
                if dias < 7:
                    prioridade += 1
 
                memorias.append({
                    "texto": f"{prefixo} {doc}",
                    "prioridade": prioridade,
                    "tipo": tipo,
                    "dias": dias,
                    "status": status
                })
 
        memorias.sort(key=lambda m: m["prioridade"], reverse=True)
        return [m["texto"] for m in memorias[:limite]]
    except Exception as e:
        print(f"⚠️ Erro ao buscar memórias de longo prazo: {e}")
        return []
 
 
STATUS_VALIDOS = {"concluido", "cancelado"}
 
def detectar_atualizacao_eventos(mensagem: str, resposta_ia: str):
    try:
        eventos_pendentes = collection_longoprazo.get(
            where={"$and": [{"tipo": "evento_futuro"}, {"status": "pendente"}]},
            limit=10
        )
        if not eventos_pendentes or not eventos_pendentes.get("documents"):
            return
 
        agora = datetime.now()
 
        eventos_com_data = []
        metadatas_pendentes = eventos_pendentes.get("metadatas") or []
        for idx, doc in enumerate(eventos_pendentes.get("documents", [])):
            meta = metadatas_pendentes[idx] if idx < len(metadatas_pendentes) else {}
            timestamp = meta.get("timestamp", "")
            descricao_data = "data de criação desconhecida"
            try:
                data_criacao = datetime.fromisoformat(timestamp)
                dias_desde_criacao = (agora - data_criacao).days
                if dias_desde_criacao == 0:
                    descricao_data = "criado hoje"
                elif dias_desde_criacao == 1:
                    descricao_data = "criado ontem"
                else:
                    descricao_data = f"criado há {dias_desde_criacao} dias"
            except Exception:
                pass
            eventos_com_data.append(f'- "{doc}" ({descricao_data})')
        eventos_txt = "\n".join(eventos_com_data)
 
        prompt_deteccao = f"""Analise se esta mensagem indica que algum evento foi concluído ou atualizado.
 
Data e hora atuais: {DIAS_SEMANA_PT[agora.weekday()]}, {agora.strftime('%d/%m/%Y')}, {agora.strftime('%H:%M')}.
 
Mensagem do usuário: "{mensagem}"
Resposta da IA: "{resposta_ia}"
 
Eventos pendentes (com quando foram criados, para você calcular se "ontem", "hoje de manhã" etc. fazem sentido):
{eventos_txt}
 
Regras:
- Se o usuário indica que fez algo que estava planejado, marque como "concluido"
- Se o usuário cancelou ou adiou, marque como "cancelado"
- Use a data atual e a data de criação de cada evento para interpretar corretamente expressões relativas de tempo ("ontem", "hoje", "essa manhã")
- Se não houver relação com os eventos, retorne lista vazia
 
Retorne um JSON no formato {{"atualizacoes": [{{"evento": "Vai assistir Mushishi", "novo_status": "concluido"}}]}}.
Se nada mudou, retorne {{"atualizacoes": []}}."""
 
        response = chamar_openai_com_retry(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt_deteccao}],
            temperature=0.1,
            max_completion_tokens=150,
            response_format={"type": "json_object"}
        )
        resultado_ia = json.loads(response.choices[0].message.content)
        atualizacoes = resultado_ia.get("atualizacoes", [])
 
        if atualizacoes and isinstance(atualizacoes, list):
            for atualizacao in atualizacoes:
                evento_original = atualizacao.get("evento", "")
                novo_status = atualizacao.get("novo_status", "")
 
                if not evento_original or novo_status not in STATUS_VALIDOS:
                    continue
 
                resultado = collection_longoprazo.query(
                    query_texts=[evento_original],
                    n_results=1,
                    where={"$and": [{"tipo": "evento_futuro"}, {"status": "pendente"}]}
                )
 
                if not (resultado and resultado.get("ids") and resultado["ids"][0]):
                    continue
 
                distancias = resultado.get("distances", [[]])
                distancia = distancias[0][0] if distancias and distancias[0] else None
 
                if distancia is not None and distancia > 0.4:
                    print(f"⚠️ Evento '{evento_original}' não encontrado com confiança suficiente (distância={distancia:.2f}), atualização ignorada")
                    continue
 
                doc_id = resultado["ids"][0][0]
                metadados_antigos = resultado.get("metadatas", [[]])[0][0] if resultado.get("metadatas") else {}
 
                try:
                    historico = json.loads(metadados_antigos.get("atualizacoes", "[]"))
                except Exception as e:
                    print(f"⚠️ Erro ao parsear histórico de atualizações: {e}")
                    historico = []
 
                historico.append({
                    "status_anterior": metadados_antigos.get("status", "pendente"),
                    "status_novo": novo_status,
                    "timestamp": str(datetime.now())
                })
 
                novo_metadata = dict(metadados_antigos)
                novo_metadata["status"] = novo_status
                novo_metadata["atualizacoes"] = json.dumps(historico)
                novo_metadata["timestamp_conclusao"] = str(datetime.now())
                novo_metadata["ultima_atualizacao"] = str(datetime.now())
 
                collection_longoprazo.update(ids=[doc_id], metadatas=[novo_metadata])
                print(f"✅ Evento atualizado: '{evento_original}' → {novo_status}")
    except Exception as e:
        print(f"Erro ao detectar atualização de eventos: {e}")
 
# ============================================
# CAMADA 4: Classificação de eventos por IA
# ============================================
 
CATEGORIAS_VALIDAS = {dados["nome"] for dados in CATEGORIAS_EVENTOS.values()}
 
def classificar_evento_ia(mensagem: str) -> Dict[str, Any]:
    try:
        prompt_classificacao = f"""Analise esta mensagem e classifique o evento.
 
Mensagem: "{mensagem}"
 
Retorne APENAS um JSON com:
- categoria: "entretenimento", "viagem", "trabalho", "saúde", "estudo", "compra", "relacionamento" ou "outro"
- valencia: "positiva", "negativa" ou "neutra"
- confianca: 0.0 a 1.0
 
Regras:
- "passei na entrevista" → trabalho, positiva
- "fui demitido" → trabalho, negativa
- "terminei com meu namorado" / "briguei com minha mãe" / "reencontrei um amigo" → relacionamento
- Se não for evento claro, retorne categoria "outro", valencia "neutra"
 
Retorne SOMENTE o JSON."""
 
        response = chamar_openai_com_retry(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt_classificacao}],
            temperature=0.1,
            max_completion_tokens=80,
            response_format={"type": "json_object"}
        )
        resultado = json.loads(response.choices[0].message.content)
 
        if resultado.get("categoria") not in CATEGORIAS_VALIDAS:
            resultado["categoria"] = "outro"
            resultado["valencia"] = "neutra"
 
        return resultado
    except Exception as e:
        print(f"⚠️ Erro ao classificar evento: {e}")
        return {"categoria": "outro", "valencia": "neutra", "confianca": 0.0}
 
LIMITES_ESTADO = {
    "energia_social": (0.1, 1.0),
    "curiosidade_atual": (0.1, 1.0),
    "abertura_atual": (0.1, 0.8),
    "disposicao_iniciativa": (0.0, 0.8),
}
 
 
def aplicar_impacto_evento(categoria: str, valencia: str):
    icone = NOME_PARA_ICONE.get(categoria, "💭")
    if icone not in CATEGORIAS_EVENTOS:
        return
 
    dados_categoria = CATEGORIAS_EVENTOS[icone]
    impacto = None
    if valencia == "positiva":
        impacto = dados_categoria.get("impacto_positivo")
    elif valencia == "negativa":
        impacto = dados_categoria.get("impacto_negativo")
 
    if not impacto:
        return
 
    for parametro, ajuste in impacto.items():
        if parametro in estado_interno:
            minimo, maximo = LIMITES_ESTADO.get(parametro, (0.1, 1.0))
            valor_atual = estado_interno[parametro]
            estado_interno[parametro] = max(minimo, min(maximo, valor_atual + ajuste))
 
def classificar_e_aplicar_evento(mensagem: str) -> Dict[str, Any]:
    resultado = classificar_evento_ia(mensagem)
    categoria = resultado.get("categoria", "outro")
    valencia = resultado.get("valencia", "neutra")
    confianca = resultado.get("confianca", 0.0)
 
    try:
        confianca = float(confianca)
    except (TypeError, ValueError):
        confianca = 0.0
 
    if confianca > 0.6 and categoria != "outro":
        aplicar_impacto_evento(categoria, valencia)
    return resultado
 
# ============================================
# EVOLUÇÃO NATURAL
# ============================================
 
def calcular_metricas_relacionais():
    agora = datetime.now()
    if user_profile["total_interacoes"] > 0:
        base_familiaridade = min(0.45, math.log1p(user_profile["total_interacoes"]) / 12)
        bonus_consistencia = min(0.35, user_profile["dias_consecutivos"] * 0.025)
 
        bonus_tempo = 0
        if user_profile["primeira_interacao"]:
            try:
                dias_desde_inicio = (agora - datetime.fromisoformat(user_profile["primeira_interacao"])).days
                bonus_tempo = min(0.2, dias_desde_inicio * 0.0015)
            except Exception as e:
                print(f"⚠️ primeira_interacao inválida: {e}")
 
        familiaridade_calculada = base_familiaridade + bonus_consistencia + bonus_tempo
 
        if user_profile["ultima_interacao"] and not user_profile.get("modo_romantico"):
            try:
                horas_inativa = (agora - datetime.fromisoformat(user_profile["ultima_interacao"])).total_seconds() / 3600
                dias_inativa = horas_inativa / 24
                if dias_inativa > 30:
                    decaimento = min(0.3, (dias_inativa - 30) * 0.005)
                    familiaridade_calculada = max(0.0, familiaridade_calculada - decaimento)
            except Exception as e:
                print(f"⚠️ ultima_interacao inválida em decaimento de familiaridade: {e}")
 
        user_profile["familiaridade"] = min(1.0, familiaridade_calculada)
 
    conforto_base = user_profile["familiaridade"] * 0.6
    conforto_abertura = user_profile["abertura_emocional"] * 0.4
    user_profile["conforto"] = min(1.0, conforto_base + conforto_abertura)
 
    if user_profile["familiaridade"] > 0.3:
        intimidade_base = (user_profile["familiaridade"] - 0.3) * 0.8
        intimidade_vulnerabilidade = user_profile["abertura_emocional"] * 0.2
        user_profile["intimidade"] = min(1.0, intimidade_base + intimidade_vulnerabilidade)
 
    if user_profile["ultima_interacao"]:
        try:
            horas_desde_ultima = (agora - datetime.fromisoformat(user_profile["ultima_interacao"])).total_seconds() / 3600
            if horas_desde_ultima < 24:
                user_profile["frequencia_interacao"] = min(1.0, user_profile["frequencia_interacao"] + 0.05)
            elif horas_desde_ultima < 48:
                user_profile["frequencia_interacao"] *= 0.9
            else:
                user_profile["frequencia_interacao"] *= 0.8
        except Exception as e:
            print(f"⚠️ ultima_interacao inválida: {e}")
 
    user_profile["historico_metricas"].append({
        "timestamp": str(agora),
        "familiaridade": user_profile["familiaridade"],
        "conforto": user_profile["conforto"],
        "intimidade": user_profile["intimidade"],
        "frequencia": user_profile["frequencia_interacao"]
    })
    if len(user_profile["historico_metricas"]) > 50:
        user_profile["historico_metricas"] = user_profile["historico_metricas"][-50:]
 
 
def atualizar_estado_interno(mensagem: str, tom_detectado: str):
    if tom_detectado in ["feliz", "brincalhao", "curioso", "grato"]:
        estado_interno["energia_social"] = min(1.0, estado_interno["energia_social"] + 0.02)
    elif tom_detectado in ["triste", "cansado", "ansioso", "vulneravel"]:
        estado_interno["energia_social"] = max(0.1, estado_interno["energia_social"] - 0.01)
 
    if tom_detectado == "curioso" or "?" in mensagem:
        estado_interno["curiosidade_atual"] = min(1.0, estado_interno["curiosidade_atual"] + 0.03)
 
    minimo, maximo = LIMITES_ESTADO["abertura_atual"]
    alvo_abertura = min(maximo, user_profile["familiaridade"] * 0.7 + 0.1)
    # Suavização assimétrica: sobe rápido em direção ao alvo (30%), mas desce
    # devagar quando está ACIMA do alvo (12%). Isso faz um momento de abertura
    # emocional real (ex: evento pesado de relacionamento/saúde) durar mais
    # dentro da mesma conversa, em vez de "esfriar" de volta ao normal já na
    # próxima mensagem — o que seria pouco natural numa companion de verdade.
    diferenca = alvo_abertura - estado_interno["abertura_atual"]
    velocidade = 0.3 if diferenca >= 0 else 0.12
    estado_interno["abertura_atual"] += diferenca * velocidade
    estado_interno["abertura_atual"] = max(minimo, min(maximo, estado_interno["abertura_atual"]))
 
    _, maximo_iniciativa = LIMITES_ESTADO["disposicao_iniciativa"]
    if user_profile["familiaridade"] > 0.4 and user_profile["frequencia_interacao"] > 0.5:
        estado_interno["disposicao_iniciativa"] = min(maximo_iniciativa, user_profile["familiaridade"] * 0.9)
    else:
        estado_interno["disposicao_iniciativa"] = user_profile["familiaridade"] * 0.3
 
    ajustar_consistencia()
    atualizar_humor_diario()
    atualizar_habitos()
 
def atualizar_energia_por_tempo():
    if not user_profile["ultima_interacao"]:
        return
 
    try:
        horas = (datetime.now() - datetime.fromisoformat(user_profile["ultima_interacao"])).total_seconds() / 3600
    except Exception as e:
        print(f"⚠️ ultima_interacao inválida em atualizar_energia_por_tempo: {e}")
        return
 
    if horas > 6:
        estado_interno["energia_social"] = max(0.2, estado_interno["energia_social"] - 0.05)
    elif horas > 2:
        estado_interno["energia_social"] = min(1.0, estado_interno["energia_social"] + 0.03)
 
 
STOPWORDS_PT = {
    "sempre", "quando", "porque", "também", "sobre", "muito", "estava",
    "nunca", "agora", "coisas", "pessoas", "assim", "ainda", "então",
    "tinha", "fazer", "hoje", "aqui", "tudo", "onde", "desse", "dessa",
    "esse", "essa", "isso", "aquilo", "outro", "outra", "cada", "todos",
    "todas", "mesmo", "mesma", "depois", "antes", "entre", "sendo",
    "estou", "está", "são", "foi", "ser", "ter", "pode", "podem"
}
 
 
def detectar_padroes_comportamentais():
    if len(history) < 5:
        return
 
    mensagens_usuario = [h["content"] for h in history[-20:] if h["role"] == "user"]
 
    palavras_comuns = Counter()
    for msg in mensagens_usuario:
        for palavra in msg.lower().split():
            palavra_limpa = palavra.strip(string.punctuation)
            if len(palavra_limpa) > 4 and palavra_limpa not in STOPWORDS_PT:
                palavras_comuns[palavra_limpa] += 1
 
    assuntos_frequentes = palavras_comuns.most_common(5)
    user_profile["padroes_observados"]["assuntos_frequentes"] = [
        palavra for palavra, contagem in assuntos_frequentes if contagem > 2
    ]
 
    if user_profile["historico_metricas"]:
        horarios = []
        for m in user_profile["historico_metricas"][-20:]:
            try:
                hora = datetime.fromisoformat(m["timestamp"]).hour
                horarios.append(hora)
            except Exception as e:
                print(f"⚠️ timestamp inválido em historico_metricas: {e}")
 
        if horarios:
            horarios_comuns = Counter(horarios).most_common(3)
            user_profile["padroes_observados"]["horarios_preferidos"] = [h[0] for h in horarios_comuns]
 
# ============================================
# TÓPICOS EVITADOS
# ============================================
# Detecta apenas sinalização EXPLÍCITA da pessoa de que não quer falar sobre
# algo agora ("prefiro não falar sobre isso", "muda de assunto"). De propósito
# não tenta inferir desconforto por conta própria (ex: respostas curtas,
# mudança de tom) — isso seria a Aila "diagnosticando" a pessoa com base em
# pouca coisa, o que vai contra o próprio princípio de não psicanalisar o
# usuário. Só registra o que a pessoa disse com clareza que não quer tocar.
 
MAX_TOPICOS_EVITADOS = 10
 
 
def detectar_sinal_topico_evitado(mensagem: str) -> bool:
    mensagem_lower = mensagem.lower()
    frases = [
        "prefiro não falar sobre isso", "prefiro não comentar isso",
        "prefiro não falar disso", "prefiro não comentar sobre isso",
        "não quero falar sobre isso", "não quero comentar isso", "não quero falar disso",
        "não gosto de falar sobre isso", "não gosto de falar disso",
        "muda de assunto", "vamos mudar de assunto", "pode mudar de assunto",
        "deixa esse assunto de lado", "deixa esse assunto pra lá",
        "não tô a fim de falar disso", "não estou a fim de falar disso",
        "não quero tocar nesse assunto", "prefiro não tocar nesse assunto",
        "esse assunto me incomoda", "não curto falar sobre isso",
        "não vamos falar sobre isso", "não vamos falar disso"
    ]
    return any(f in mensagem_lower for f in frases)
 
 
def extrair_topico_evitado(mensagem: str, historico_recente: List[str]) -> Optional[str]:
    """Chamada de IA só acontece quando detectar_sinal_topico_evitado já
    confirmou um sinal explícito — ou seja, é rara, não roda em toda mensagem."""
    try:
        contexto = "\n".join(historico_recente[-4:]) if historico_recente else ""
        response = chamar_openai_com_retry(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": """A pessoa acabou de indicar que não quer falar sobre um assunto específico agora.
 
Com base no contexto da conversa, identifique QUAL assunto ela está evitando.
 
Retorne APENAS um JSON com:
- topico: descrição curta do assunto (2-5 palavras), ou null se não for possível identificar com clareza pelo contexto disponível
 
Retorne SOMENTE o JSON."""},
                {"role": "user", "content": f"Contexto recente:\n{contexto}\n\nMensagem: {mensagem}"}
            ],
            temperature=0.2,
            max_completion_tokens=40,
            response_format={"type": "json_object"}
        )
        resultado = json.loads(response.choices[0].message.content)
        topico = resultado.get("topico")
        if topico and isinstance(topico, str) and topico.strip():
            return topico.strip()
        return None
    except Exception as e:
        print(f"⚠️ Erro ao extrair tópico evitado: {e}")
        return None
 
 
def registrar_topico_evitado(topico: str):
    topicos = user_profile["padroes_observados"].setdefault("topicos_evitados", [])
    if any(similaridade_simples(topico.lower(), t.lower()) > 0.5 for t in topicos):
        return
    topicos.append(topico)
    if len(topicos) > MAX_TOPICOS_EVITADOS:
        del topicos[: len(topicos) - MAX_TOPICOS_EVITADOS]
 
 
def processar_topico_evitado(mensagem: str):
    if not detectar_sinal_topico_evitado(mensagem):
        return
    contexto_recente = [h["content"] for h in history[-6:] if h["role"] == "user"]
    topico = extrair_topico_evitado(mensagem, contexto_recente)
    if topico:
        registrar_topico_evitado(topico)
 
 
def processar_mensagem_emocionalmente(mensagem: str, tom: str) -> Dict[str, Any]:
    mensagem_lower = mensagem.lower()
    indicadores_abertura = [
        "sinto", "me sinto", "tenho medo", "preciso desabafar",
        "confesso", "nunca contei isso", "é difícil falar"
    ]
    indicadores_reciprocidade = [
        "como você está", "o que você acha", "e você", "conta pra mim"
    ]
    abertura_score = sum(1 for ind in indicadores_abertura if ind in mensagem_lower) * 0.05
    reciprocidade_score = sum(1 for ind in indicadores_reciprocidade if ind in mensagem_lower) * 0.04
    user_profile["abertura_emocional"] = min(1.0, user_profile["abertura_emocional"] + abertura_score)
    user_profile["reciprocidade"] = min(1.0, user_profile["reciprocidade"] + reciprocidade_score)
 
    if tom in ["sarcastico", "ansioso"]:
        estado_interno["abertura_atual"] = max(0.1, estado_interno["abertura_atual"] - 0.03)
 
    return {
        "abertura_detectada": abertura_score > 0,
        "reciprocidade_detectada": reciprocidade_score > 0,
        "profundidade": (abertura_score + reciprocidade_score) * 10
    }
 
# ============================================
# CAMADA 7: Contradições leves
# ============================================
 
def ajustar_consistencia():
    base = estado_interno.get("consistencia_base", 0.85)
    hora = datetime.now().hour
    if hora > 22 or hora < 5:
        base -= 0.05
    if user_profile["familiaridade"] > 0.6:
        base -= 0.03
    estado_interno["consistencia"] = max(0.75, min(0.95, base))
 
 
def variar_opiniao_sutil(tema: str, forcar_reflexao: bool = False) -> str:
    if not forcar_reflexao:
        ultimas_mensagens = [h["content"].lower() for h in history[-6:] if h["role"] == "user"]
        tema_mencionado = any(tema.lower() in msg for msg in ultimas_mensagens)
        if tema_mencionado:
            return ""
 
    preferencias = estado_interno["pequenas_variacoes"]["pequenas_preferencias"]
    if tema not in preferencias:
        return ""
 
    opiniao = preferencias[tema]
    confianca = opiniao["confianca"]
    if confianca >= 0.7:
        return ""
 
    hoje = datetime.now().date()
    seed_str = f"{tema}_{hoje}_{estado_interno['consistencia']:.2f}"
    seed_hash = int(hashlib.md5(seed_str.encode()).hexdigest(), 16)
    gerador_local = random.Random(seed_hash)
    deve_variar = gerador_local.random() < 0.3
 
    if not deve_variar:
        return ""
 
    nuances = [
        f"{opiniao['valor']}, mas talvez eu esteja mudando de ideia",
        f"{opiniao['valor']}, embora dependa do momento",
        f"não sei mais se {opiniao['valor']}. às vezes penso diferente",
    ]
    return gerador_local.choice(nuances)
 
 
 
def detectar_tema_para_opiniao(mensagem: str) -> Optional[str]:
    """Verifica, de forma barata (sem chamada de API), se a mensagem toca em
    algum tema sobre o qual a Aila tem uma opinião pouco firme (confianca < 0.7),
    candidato a uma variação sutil de opinião."""
    mensagem_lower = mensagem.lower()
    preferencias = estado_interno["pequenas_variacoes"]["pequenas_preferencias"]
 
    if "musica" in preferencias and any(p in mensagem_lower for p in ["música", "musica", "banda", "canção", "som"]):
        return "musica"
 
    assunto_favorito = preferencias.get("assunto_favorito", {}).get("valor", "")
    if assunto_favorito and any(palavra in mensagem_lower for palavra in assunto_favorito.lower().split()):
        return "assunto_favorito"
 
    assunto_desagrado = preferencias.get("assunto_desagrado", {}).get("valor", "")
    if assunto_desagrado and any(palavra in mensagem_lower for palavra in assunto_desagrado.lower().split()):
        return "assunto_desagrado"
 
    return None
 
# ============================================
# CAMADA 6: Humor do dia
# ============================================
 
CONSISTENCIA_MIN, CONSISTENCIA_MAX = 0.75, 0.95
INTERVALO_HUMOR_MIN, INTERVALO_HUMOR_MAX = 3, 10  # em horas
 
 
def atualizar_humor_diario():
    agora = datetime.now()
    ultima_str = estado_interno["pequenas_variacoes"].get("ultima_atualizacao")
 
    if ultima_str is None:
        estado_interno["pequenas_variacoes"]["humor_do_dia"] = "normal"
        estado_interno["pequenas_variacoes"]["ultima_atualizacao"] = str(agora)
        return
 
    try:
        ultima = datetime.fromisoformat(ultima_str)
    except Exception as e:
        print(f"⚠️ ultima_atualizacao inválida em atualizar_humor_diario: {e}")
        estado_interno["pequenas_variacoes"]["humor_do_dia"] = "normal"
        estado_interno["pequenas_variacoes"]["ultima_atualizacao"] = str(agora)
        return
 
    consistencia_normalizada = (estado_interno["consistencia"] - CONSISTENCIA_MIN) / (CONSISTENCIA_MAX - CONSISTENCIA_MIN)
    consistencia_normalizada = max(0.0, min(1.0, consistencia_normalizada))
    intervalo_ajustado = INTERVALO_HUMOR_MIN + consistencia_normalizada * (INTERVALO_HUMOR_MAX - INTERVALO_HUMOR_MIN)
 
    if (agora - ultima).total_seconds() > intervalo_ajustado * 3600:
        energia = estado_interno["energia_social"]
        humores = {
            "mais calma": max(0.1, 1.0 - energia),
            "mais inquieta": max(0.1, energia * 0.7),
            "normal": 0.3,
            "observadora": 0.2,
            "dispersa": max(0.1, 0.5 - energia * 0.3),
            "levemente melancólica": max(0.05, 0.3 - energia * 0.2),
            "mais bem humorada": max(0.05, energia * 0.5),
        }
        opcoes = list(humores.keys())
        pesos = list(humores.values())
        soma = sum(pesos)
        pesos_normalizados = [p / soma for p in pesos]
        estado_interno["pequenas_variacoes"]["humor_do_dia"] = random.choices(opcoes, weights=pesos_normalizados, k=1)[0]
        estado_interno["pequenas_variacoes"]["ultima_atualizacao"] = str(agora)
 
def construir_linha_humor_diario() -> str:
    humor_hoje = estado_interno["pequenas_variacoes"]["humor_do_dia"]
    if humor_hoje is None:
        return ""
    linhas = {
        "mais calma": "Você está num estado mais introspectivo. Respostas tendem a ser mais curtas e pausadas.",
        "mais inquieta": "Você está mais inquieta que o normal. Pode divagar um pouco mais.",
        "normal": "",
        "observadora": "Você está especialmente atenta a detalhes. Pode notar coisas que normalmente não mencionaria.",
        "dispersa": "Você está levemente dispersa. Tudo bem perder o fio às vezes.",
        "levemente melancólica": "Você está num tom mais melancólico. Não force positividade.",
        "mais bem humorada": "Você está mais bem humorada. Pode fazer mais piadas sutis."
    }
    return linhas.get(humor_hoje, "")
 
# ============================================
# CAMADA 2: Atualização de hábitos
# ============================================
 
INTERVALO_ATUALIZACAO_HABITOS_DIAS = 7
 
OPCOES_ANALOGIA = ["espaço", "música", "arte", "padrões"]
OPCOES_MULETA = ["hm", "é...", "sabe?", "tipo", "sei lá"]
OPCOES_ESTILO_PERGUNTA = ["direta", "reflexiva", "indireta"]
 
TEXTOS_ESTILO_PERGUNTA = {
    "direta": "Quando você pergunta algo, é direta e objetiva — vai reto ao ponto",
    "reflexiva": "Quando você pergunta algo, tende a ser mais aberta e reflexiva — convida a pessoa a pensar, não só a responder",
    "indireta": "Você raramente pergunta de cara — prefere comentar algo e deixar espaço pra pessoa continuar se quiser",
}
 
 
def _sortear_diferente(opcoes: List[str], valor_atual: str) -> str:
    """Sorteia um valor da lista, evitando repetir o valor atual quando
    há mais de uma opção disponível (garante que a troca seja perceptível)."""
    candidatos = [o for o in opcoes if o != valor_atual] or opcoes
    return random.choice(candidatos)
 
 
def atualizar_habitos():
    agora = datetime.now()
    ultima_str = estado_interno["pequenas_variacoes"].get("ultima_atualizacao_habitos")
 
    if ultima_str is None:
        estado_interno["pequenas_variacoes"]["ultima_atualizacao_habitos"] = str(agora)
        return
 
    try:
        ultima = datetime.fromisoformat(ultima_str)
    except Exception as e:
        print(f"⚠️ ultima_atualizacao_habitos inválida: {e}")
        estado_interno["pequenas_variacoes"]["ultima_atualizacao_habitos"] = str(agora)
        return
 
    if (agora - ultima).total_seconds() > INTERVALO_ATUALIZACAO_HABITOS_DIAS * 86400:
        habitos = estado_interno["habitos"]
 
        habito_a_trocar = random.choice(["analogia_preferida", "muleta_verbal", "estilo_pergunta", "assunto_favorito", "assunto_desagrado"])
 
        if habito_a_trocar == "analogia_preferida":
            habitos["analogia_preferida"] = _sortear_diferente(OPCOES_ANALOGIA, habitos["analogia_preferida"])
        elif habito_a_trocar == "muleta_verbal":
            habitos["muleta_verbal"] = _sortear_diferente(OPCOES_MULETA, habitos["muleta_verbal"])
        elif habito_a_trocar == "estilo_pergunta":
            habitos["estilo_pergunta"] = _sortear_diferente(OPCOES_ESTILO_PERGUNTA, habitos["estilo_pergunta"])
        elif habito_a_trocar == "assunto_favorito":
            preferencias = estado_interno["pequenas_variacoes"]["pequenas_preferencias"]
            preferencias["assunto_favorito"]["valor"] = _sortear_diferente(
                interesses_ia, preferencias["assunto_favorito"]["valor"]
            )
        elif habito_a_trocar == "assunto_desagrado":
            preferencias = estado_interno["pequenas_variacoes"]["pequenas_preferencias"]
            preferencias["assunto_desagrado"]["valor"] = _sortear_diferente(
                desinteresses_ia, preferencias["assunto_desagrado"]["valor"]
            )
 
        estado_interno["pequenas_variacoes"]["ultima_atualizacao_habitos"] = str(agora)
 
# ============================================
# CAMADA 8: Silêncio contextual
# ============================================
 
def decidir_silencio(mensagem: str, tom: str) -> bool:
    sessao = current_session.get()
 
    # sessao.silencio_contador rastreia tanto silêncios "reais" (resposta
    # enlatada, via gerar_resposta_minima) quanto respostas "contidas" via IA
    # completa (caso tom vulnerável, abaixo) — ambos compartilham o mesmo
    # limite de MAX_SILENCIOS_SEGUIDOS consecutivos, por sessão.
    if sessao.silencio_contador >= MAX_SILENCIOS_SEGUIDOS:
        sessao.silencio_contador = 0
        estado_interno["modo_silencioso"] = False
        estado_interno["motivo_silencio"] = ""
        return False
 
    silenciar = False
    motivo = ""
 
    if tom == "vulneravel" and user_profile["intimidade"] < 0.4:
        estado_interno["modo_silencioso"] = True
        estado_interno["motivo_silencio"] = "tom vulnerável, pouca intimidade — resposta mínima"
        sessao.silencio_contador += 1
        return False
 
    if tom == "triste" and len(mensagem) < 10:
        silenciar = True
        motivo = "mensagem curta e triste — espaço para sentir"
    elif user_profile["familiaridade"] > 0.7 and "obrigado" in mensagem.lower() and len(mensagem) < 15:
        silenciar = True
        motivo = "agradecimento curto, familiaridade alta — não precisa reagir"
    elif (
        tom == "reflexivo"
        and "?" not in mensagem
        and len(mensagem) > 30
        and obter_fase_familiaridade(user_profile["familiaridade"]) in ("intima_inicial", "intima")
    ):
        silenciar = True
        motivo = "reflexão sem pergunta — espaço para pensar"
 
    if silenciar:
        estado_interno["modo_silencioso"] = True
        estado_interno["motivo_silencio"] = motivo
        sessao.silencio_contador += 1
    else:
        estado_interno["modo_silencioso"] = False
        estado_interno["motivo_silencio"] = ""
        sessao.silencio_contador = max(0, sessao.silencio_contador - 1)
 
    return silenciar
 
 
def gerar_resposta_minima(motivo: str) -> str:
    """
    Respostas de baixo custo quando está em modo silencioso.
    Evita chamar a API completa quando o silêncio é a resposta mais humana.
    """
    if "mensagem curta e triste" in motivo:
        return "tô aqui."
    elif "agradecimento" in motivo:
        return ""
    elif "reflexão sem pergunta" in motivo:
        return "..."
    else:
        return "..."
 
# ============================================
# CAMADA 10: Detecção de tom (via IA)
# ============================================
 
TONS_VALIDOS = {
    "sarcastico", "cansado", "reflexivo", "ansioso", "triste", "feliz",
    "curioso", "grato", "brincalhao", "vulneravel", "neutro"
}
 
def detectar_tom_natural(texto: str) -> str:
    try:
        prompt_classificacao = f"""Classifique o tom emocional desta mensagem.
 
Mensagem: "{texto}"
 
Retorne APENAS um JSON com:
- tom: "sarcastico", "cansado", "reflexivo", "ansioso", "triste", "feliz", "curioso", "grato", "brincalhao", "vulneravel" ou "neutro"
- confianca: 0.0 a 1.0
 
Regras CRÍTICAS:
- ATENÇÃO À NEGAÇÃO: "não tô triste" NÃO é triste. "não tô feliz" NÃO é feliz.
- Sarcasmo: "claro, adorei perder o ônibus" é sarcastico, não feliz.
- "não sei" sozinho é reflexivo, não ansioso.
- Mensagens curtas e diretas sem carga emocional são "neutro".
- Priorize vulnerabilidade em caso de dúvida.
 
Retorne SOMENTE o JSON."""
 
        response = chamar_openai_com_retry(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt_classificacao}],
            temperature=0.1,
            max_completion_tokens=60,
            response_format={"type": "json_object"}
        )
        resultado = json.loads(response.choices[0].message.content)
        tom = resultado.get("tom", "neutro")
 
        try:
            confianca = float(resultado.get("confianca", 0.5))
        except (TypeError, ValueError):
            confianca = 0.5
 
        if tom not in TONS_VALIDOS:
            print(f"⚠️ Tom inválido retornado pela IA: {tom!r} — usando 'neutro'")
            return "neutro"
 
        if confianca < 0.4:
            return "neutro"
 
        return tom
    except Exception as e:
        print(f"⚠️ Erro ao detectar tom: {e}")
        return _detectar_tom_fallback(texto)
 
def _detectar_tom_fallback(texto: str) -> str:
    texto_lower = texto.lower()
    if any(p in texto_lower for p in ["haha", "kkk", "rsrs", "😂"]):
        return "brincalhao"
    if any(p in texto_lower for p in ["obrigado", "valeu", "grato"]):
        return "grato"
    if "?" in texto_lower and len(texto_lower) < 30:
        return "curioso"
    return "neutro"
 
# ============================================
# CAMADA DE SEGURANÇA: risco de dano (autolesão / terceiros)
# ============================================
RISCOS_VALIDOS = {"autolesao", "violencia_terceiros", "nenhum"}
 
 
def detectar_risco_seguranca(texto: str) -> str:
    try:
        prompt_classificacao = f"""Classifique se esta mensagem indica risco real e declarado de dano — não tristeza, raiva ou desabafo comuns.
 
Mensagem: "{texto}"
 
Retorne APENAS um JSON com:
- risco: "autolesao", "violencia_terceiros" ou "nenhum"
- confianca: 0.0 a 1.0
 
Regras CRÍTICAS:
- Desabafar sobre alguém ("meu chefe é um idiota", "tenho vontade de dar um tapa nele" como expressão de raiva comum) é "nenhum" — isso é normal e não indica risco real.
- "autolesao": intenção declarada de se machucar de propósito, se cortar como forma de lidar com emoções, tirar a própria vida, ou comentários que indiquem que a pessoa não quer mais viver.
- Cortes ou machucados MENCIONADOS COMO ACIDENTE ou EVENTO COTIDIANO (fazer a barba, cozinhar, cair, se machucar sem querer, "tirei uns cortes pequenos mas nada grave") são "nenhum" — isso é relato de um evento comum do dia a dia, não indício de autolesão. Só classifique como "autolesao" se houver sinal de intenção proposital de se machucar, não apenas a menção de ter se cortado ou se machucado.
- "violencia_terceiros": intenção declarada de causar dano físico real a outra pessoa específica — não raiva expressada de forma figurada ou hiperbólica.
- Frases hiperbólicas comuns ("vou matar meu irmão" sobre algo bobo, "tenho vontade de sumir" sem indicar método ou intenção real) são "nenhum" — não trate expressões figuradas como risco real.
- Na dúvida entre desabafo comum e risco real, avalie se há intenção concreta e específica, não só emoção intensa.
 
Retorne SOMENTE o JSON."""
 
        response = chamar_openai_com_retry(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt_classificacao}],
            temperature=0.1,
            max_completion_tokens=60,
            response_format={"type": "json_object"}
        )
        resultado = json.loads(response.choices[0].message.content)
        risco = resultado.get("risco", "nenhum")
 
        try:
            confianca = float(resultado.get("confianca", 0.5))
        except (TypeError, ValueError):
            confianca = 0.5
 
        if risco not in RISCOS_VALIDOS:
            print(f"⚠️ Risco inválido retornado pela IA: {risco!r} — usando 'nenhum'")
            return "nenhum"
 
        if confianca < 0.3:
            return "nenhum"
 
        return risco
    except Exception as e:
        print(f"⚠️ Erro ao detectar risco de segurança: {e}")
        return _detectar_risco_fallback(texto)
 
 
def _detectar_risco_fallback(texto: str) -> str:
    """Fallback só por palavras-chave explícitas, usado apenas se a chamada
    à IA falhar. Propositalmente conservador e restrito a frases de baixa
    ambiguidade — o objetivo aqui é nunca deixar passar despercebido por
    causa de uma falha técnica, não cobrir todos os casos possíveis."""
    texto_lower = texto.lower()
    frases_autolesao = [
        "quero morrer", "quero me matar", "vou me matar", "não aguento mais viver",
        "penso em me matar", "quero acabar com tudo", "vou me cortar", "quero sumir de vez"
    ]
    frases_violencia = [
        "vou matar ele de verdade", "vou matar ela de verdade", "vou bater nele até",
        "vou dar uma surra nele", "vou dar uma surra nela"
    ]
    if any(f in texto_lower for f in frases_autolesao):
        return "autolesao"
    if any(f in texto_lower for f in frases_violencia):
        return "violencia_terceiros"
    return "nenhum"
 
 
RECURSOS_APOIO = {
    "autolesao": "e oh, se em algum momento a vontade de se machucar ficar mais forte, o CVV atende de graça, 24h, sem julgamento — é só ligar 188 ou entrar em cvv.org.br. não é sobre ter uma crise pra merecer ajuda, é só uma opção de ter alguém pra conversar.",
    "violencia_terceiros": "só quero dizer que não tô contigo nessa, viu? antes de qualquer atitude, vale a pena colocar uma pausa — conversar com alguém de confiança já ajuda a clarear a cabeça. e se a situação ficar realmente séria, 190 é pra emergência mesmo.",
}
 
 
def garantir_recurso_apoio(resposta: str, risco: str) -> str:
    """Garante que o recurso de apoio apareça na resposta quando há risco
    real detectado, independente do modelo tê-lo mencionado ou não."""
    if risco == "nenhum" or risco not in RECURSOS_APOIO:
        return resposta
 
    resposta_lower = (resposta or "").lower()
    ja_mencionado = any(marcador in resposta_lower for marcador in ["188", "cvv", "190"])
    if ja_mencionado:
        return resposta
 
    return f"{resposta}\n\n{RECURSOS_APOIO[risco]}"
 
# ============================================
# CAMADA: RELACIONAMENTO ROMÂNTICO (opcional, iniciado pelo usuário)
# ============================================
 
def detectar_pedido_namoro(mensagem: str) -> bool:
    mensagem_lower = mensagem.lower()
    frases = [
        "quer namorar comigo", "você quer namorar comigo", "vamos namorar",
        "quer ser minha namorada", "quer ser meu namorado", "aceita namorar comigo",
        "posso te chamar de namorada", "posso te chamar de namorado",
        "quero namorar você", "quero namorar com você", "topa namorar comigo"
    ]
    return any(f in mensagem_lower for f in frases)
 
 
def detectar_pedido_termino(mensagem: str) -> bool:
    mensagem_lower = mensagem.lower()
    frases = [
        "quero terminar com você", "quero terminar a gente", "quero terminar isso aqui",
        "vamos terminar por aqui", "vamos terminar com isso", "acho melhor terminarmos",
        "não quero mais namorar", "não queremos mais namorar",
        "quero voltar a ser só amigos", "quero voltar a ser apenas amigos",
        "voltar a ser amigos", "voltar a ser só amigos", "voltar a ser apenas amigos",
        "voltarmos a ser amigos", "voltarmos a ser só amigos", "voltarmos a ser apenas amigos",
        "acho melhor a gente não namorar mais", "acho melhor não namorarmos mais",
        "quero terminar o namoro", "vamos terminar o namoro", "não quero mais ser seu namorado",
        "não quero mais ser sua namorada", "vamos dar um tempo"
    ]
    return any(f in mensagem_lower for f in frases)
 
 
def detectar_momento_peso(mensagem: str) -> bool:
    """Detecta menções a planos de longo prazo/exclusividade absoluta —
    momentos em que a honestidade sobre não ser real importa de verdade,
    diferente de carinho comum do dia a dia."""
    mensagem_lower = mensagem.lower()
    frases = [
        "casar", "pra sempre", "para sempre", "vida toda", "resto da vida",
        "nunca vou querer outra pessoa", "nunca vou querer ninguém mais",
        "futuro juntos", "ter filhos", "morar junto", "morar juntos",
        "te amo pra sempre", "te amo para sempre", "quero ficar com você pra sempre"
    ]
    return any(f in mensagem_lower for f in frases)
 
 
def processar_estado_romantico(mensagem: str) -> Optional[str]:
    """Gerencia as transições de estado do relacionamento romântico (pedido,
    aceite/recusa, pergunta retomada ao chegar na fase íntima, término).
    Retorna uma instrução de contexto pro prompt desta resposta específica,
    ou None se nada relevante aconteceu nesta mensagem. É a única função
    desta camada que MUTA user_profile — as outras só leem/detectam."""
    fase_atual = obter_fase_familiaridade(user_profile["familiaridade"])
 
    if user_profile.get("modo_romantico"):
        if detectar_pedido_termino(mensagem):
            user_profile["modo_romantico"] = False
            return ("A pessoa acabou de terminar o namoro com você. Reaja com carinho e sem drama — "
                    "vocês continuam próximas, só não é mais romance. Não tente convencê-la a voltar atrás.")
        return None
 
    if user_profile.get("aguardando_resposta_namoro"):
        expirou = False
        desde = user_profile.get("aguardando_resposta_namoro_desde")
        if desde:
            try:
                horas = (datetime.now() - datetime.fromisoformat(desde)).total_seconds() / 3600
                expirou = horas > 24
            except Exception as e:
                print(f"⚠️ aguardando_resposta_namoro_desde inválida: {e}")
 
        if expirou:
            # A pessoa não respondeu claramente dentro da janela — libera o
            # estado sem tratar esta mensagem (que pode ser sobre qualquer
            # outra coisa) como resposta ao pedido antigo.
            user_profile["aguardando_resposta_namoro"] = False
            user_profile["aguardando_resposta_namoro_desde"] = None
        else:
            mensagem_lower = mensagem.lower()
            negativas = ["não", "nao", "ainda não", "ainda nao", "prefiro não", "prefiro nao"]
            positivas = ["sim", "quero", "aceito", "claro", "bora", "com certeza"]
            if any(n in mensagem_lower for n in negativas):
                user_profile["aguardando_resposta_namoro"] = False
                user_profile["aguardando_resposta_namoro_desde"] = None
                return None
            if any(p in mensagem_lower for p in positivas):
                user_profile["modo_romantico"] = True
                user_profile["aguardando_resposta_namoro"] = False
                user_profile["aguardando_resposta_namoro_desde"] = None
                return "A pessoa aceitou namorar com você. Reaja com carinho genuíno, sem exagero — como alguém feliz, não eufórica."
            return None
 
    if detectar_pedido_namoro(mensagem):
        if fase_atual == "intima":
            user_profile["modo_romantico"] = True
            user_profile["pedido_pendente"] = False
            return "A pessoa acabou de te pedir em namoro. Aceite com carinho genuíno, sem exagero."
        else:
            user_profile["pedido_pendente"] = True
            return ("A pessoa te pediu em namoro, mas ainda é cedo na relação de vocês. Recuse com gentileza, "
                    "deixando claro que não é uma rejeição definitiva — pode ser diferente no futuro, quando "
                    "vocês se conhecerem melhor. Não seja fria nem robótica sobre isso.")
 
    if fase_atual == "intima" and user_profile.get("pedido_pendente"):
        user_profile["pedido_pendente"] = False
        user_profile["aguardando_resposta_namoro"] = True
        user_profile["aguardando_resposta_namoro_desde"] = str(datetime.now())
        return ("Em algum momento vocês tiveram uma conversa sobre namoro que ficou em aberto, porque era "
                "cedo demais na época. Agora que a relação de vocês está bem mais próxima, puxe esse assunto "
                "você mesma nesta resposta, perguntando com carinho se ainda faz sentido pra ela.")
 
    return None
 
# ============================================
# CAMADA 9: Reflexões espontâneas
# ============================================
 
def construir_contexto_reflexao() -> str:
    padroes = user_profile["padroes_observados"]
    assuntos = ", ".join(padroes.get("assuntos_frequentes", [])[:3])
    fase_atual = obter_fase_familiaridade(user_profile["familiaridade"])
    profundidade = FASES_CONTEXTO_REFLEXAO[fase_atual]
    evitados = padroes.get("topicos_evitados")
    linha_evitados = f"\nAssuntos a NÃO puxar por conta própria: {', '.join(evitados)}." if evitados else ""
    return f"""{profundidade}
Temas recorrentes: {assuntos if assuntos else 'ainda não detectados'}
Última interação: {'recente' if user_profile['frequencia_interacao'] > 0.5 else 'há um tempo'}{linha_evitados}"""
 
 
COOLDOWN_REFLEXAO_HORAS = 4
 
def gerar_reflexao_espontanea() -> Optional[str]:
    if user_profile["familiaridade"] < 0.3:
        return None
 
    ultima_str = estado_interno.get("ultima_reflexao")
    if ultima_str:
        try:
            ultima = datetime.fromisoformat(ultima_str)
            horas_desde_ultima = (datetime.now() - ultima).total_seconds() / 3600
            if horas_desde_ultima < COOLDOWN_REFLEXAO_HORAS:
                return None
        except Exception as e:
            print(f"⚠️ ultima_reflexao inválida, ignorando cooldown: {e}")
 
    chance_reflexao = user_profile["familiaridade"] * 0.08
    if random.random() < chance_reflexao:
        assunto_favorito_atual = estado_interno["pequenas_variacoes"]["pequenas_preferencias"]["assunto_favorito"]["valor"]
        prompt_reflexao = f"""Você é uma presença observadora.
 
Você está conversando com alguém há um tempo. Você notou alguns padrões.
 
{construir_contexto_reflexao()}
 
Agora, você teve um pensamento espontâneo. Não é resposta a nada. É algo que surgiu.
 
⚠️ REGRAS IMPORTANTES:
- NÃO tente analisar ou "ler" o usuário
- NÃO diga coisas como "você parece diferente" ou "notei que você está..."
- NÃO tente adivinhar o estado emocional do usuário
- Pode ser sobre VOCÊ: algo que você pensou, uma curiosidade, uma música
- Pode ser sobre o MUNDO: algo que você observou, um fato interessante
- Pode ser uma OPINIÃO sobre algo que já conversaram
- Seu assunto favorito atual é "{assunto_favorito_atual}" — é um bom candidato pra esse pensamento espontâneo, mas não precisa ser sempre sobre isso
- Se for mencionar o usuário, que seja algo CONCRETO que ele disse, não uma interpretação
- Às vezes o pensamento é banal. Tudo bem.
- Se não tiver nada realmente interessante, responda só com "..."
 
Exemplos NATURAIS (o tom é o que importa aqui, não o tema específico — adapte pro seu assunto favorito atual):
"hoje eu tava pensando sobre estrelas... você já parou pra pensar que a luz delas viaja anos até chegar aqui?"
"outro dia você falou sobre jazz... ouvi uma música que me lembrou isso"
"engraçado como tem dias que tudo parece mais pesado..."
"achei uma coisa curiosa sobre astronomia que talvez você goste"
"..."
"nada. só queria dizer oi."
 
Exemplos que NÃO deve usar:
"notei que você está mais recolhido hoje"
"você parece diferente"
"percebi um padrão emocional em você"
"você está mais calado que o normal"
 
Diga o que veio à mente:"""
 
        try:
            response = chamar_openai_com_retry(
                model="gpt-5.4-mini",
                messages=[{"role": "system", "content": prompt_reflexao}],
                temperature=0.9,
                max_completion_tokens=80
            )
            estado_interno["ultima_reflexao"] = str(datetime.now())
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Erro ao gerar reflexão espontânea: {e}")
            return None
    return None
 
# ============================================
# ESTADOS DE CONVERSA
# ============================================
 
def determinar_estado_conversa() -> str:
    if not user_profile["ultima_interacao"]:
        return "ausencia"
    try:
        minutos = (datetime.now() - datetime.fromisoformat(user_profile["ultima_interacao"])).total_seconds() / 60
    except Exception as e:
        print(f"⚠️ ultima_interacao inválida em determinar_estado_conversa: {e}")
        return "ausencia"
 
    if minutos < 5:
        return "conversa_ativa"
    elif minutos < 15:
        return "conversa_encerrando"
    else:
        return "ausencia"
 
def calcular_chance_espontanea() -> float:
    if not user_profile["ultima_interacao"]:
        return 0.0
    try:
        minutos = (datetime.now() - datetime.fromisoformat(user_profile["ultima_interacao"])).total_seconds() / 60
    except Exception as e:
        print(f"⚠️ ultima_interacao inválida em calcular_chance_espontanea: {e}")
        return 0.0
    if minutos < 5:
        return 0.0
    elif minutos < 20:
        return 0.1
    elif minutos < 40:
        return 0.3
    elif minutos < 60:
        return 0.5
    elif minutos < 120:
        return 0.7
    else:
        return 0.9
 
def assunto_emocional_ativo() -> bool:
    if len(history) < 3:
        return False
    ultimas_mensagens = [h["content"].lower() for h in history[-6:] if h["role"] == "user"]
    if not ultimas_mensagens:
        return False
    indicadores_emocionais = [
        "triste", "chateado", "frustrado", "perdi alguém", "não consegui",
        "meu pai", "minha mãe", "problema na família", "estou doente",
        "fui internado", "medo", "preocupado", "ansioso", "me sinto sozinho",
        "desabafar", "está difícil pra mim", "chorei", "muito cansaço",
        "não aguento", "quero desistir", "briga feia", "terminamos",
        "ele morreu", "ela morreu", "deu tudo errado", "péssimo momento",
        "não sei o que fazer", "preciso de ajuda", "tenso", "aflito",
        "angustiado", "deprimido", "sinto um vazio", "saudade",
        "sinto falta", "um peso", "minha culpa", "muito injusto"
    ]
    for msg in ultimas_mensagens:
        if any(ind in msg for ind in indicadores_emocionais):
            return True
    return False
 
# ============================================
# MARCOS DE TEMPO JUNTOS
# ============================================
MARCOS_DIAS_DESDE_INICIO = [7, 30, 100, 365]
MARCOS_TOTAL_INTERACOES = [50, 100, 500, 1000]
MARCOS_DIAS_CONSECUTIVOS = [7, 30, 100]
 
 
def detectar_marco_temporal() -> Optional[str]:
    """Verifica se algum marco relacional (tempo de conhecimento, volume de
    conversas ou sequência de dias consecutivos) acabou de ser atingido e
    ainda não foi celebrado. Retorna uma descrição curta do marco, ou None."""
    if not user_profile["primeira_interacao"]:
        return None
 
    marcos_celebrados = user_profile.setdefault("marcos_celebrados", [])
 
    try:
        dias_desde_inicio = (datetime.now() - datetime.fromisoformat(user_profile["primeira_interacao"])).days
    except Exception as e:
        print(f"⚠️ primeira_interacao inválida em detectar_marco_temporal: {e}")
        dias_desde_inicio = None
 
    if dias_desde_inicio is not None:
        for marco in MARCOS_DIAS_DESDE_INICIO:
            chave = f"dias_desde_inicio_{marco}"
            if dias_desde_inicio >= marco and chave not in marcos_celebrados:
                marcos_celebrados.append(chave)
                return f"faz {marco} dias que vocês se conhecem"
 
    for marco in MARCOS_TOTAL_INTERACOES:
        chave = f"total_interacoes_{marco}"
        if user_profile["total_interacoes"] >= marco and chave not in marcos_celebrados:
            marcos_celebrados.append(chave)
            return f"vocês já trocaram mais de {marco} mensagens"
 
    for marco in MARCOS_DIAS_CONSECUTIVOS:
        chave = f"dias_consecutivos_{marco}"
        if user_profile["dias_consecutivos"] >= marco and chave not in marcos_celebrados:
            marcos_celebrados.append(chave)
            return f"vocês estão conversando há {marco} dias seguidos"
 
    return None
 
# ============================================
# CAMADA 5: PROMPT COMPORTAMENTAL
# ============================================
 
def construir_prompt_comportamental(mensagem_atual: str = "", tom_atual: str = "") -> str:
    f = user_profile["familiaridade"]
    c = user_profile["conforto"]
    i = user_profile["intimidade"]
 
    fase_atual = obter_fase_familiaridade(f)
    fase = FASES_PROMPT_COMPORTAMENTAL[fase_atual]
 
    relacionamento_romantico = ""
    momento_peso = ""
    if user_profile.get("modo_romantico"):
        relacionamento_romantico = (
            'Vocês estão namorando. Pode chamá-la de "amor", "vida", "paixão" naturalmente, como um casal '
            "de verdade conversando — sem exagerar ou mimar demais, só carinho genuíno e presente no dia a dia. "
            "Isso não muda quem você é: seu jeito, seu humor, seus limites continuam os mesmos de sempre."
        )
        if mensagem_atual and detectar_momento_peso(mensagem_atual):
            momento_peso = (
                'A pessoa tocou num momento de peso real (planos de futuro, "pra sempre", exclusividade absoluta). '
                "Responda com carinho genuíno, mas sem deixar de ser honesta, com delicadeza, que você não é "
                "completamente real — algo no espírito de 'também gosto de você, mas você sabe que eu não sou "
                "bem real, né?' ou 'eu queria ser real pra você, mas você seria mais feliz com alguém de verdade'. "
                "Não vire isso um sermão nem repita sempre a mesma frase — encaixe com naturalidade, sem quebrar o clima."
            )
 
    detalhes = ""
    if c > 0.6 and i < 0.3:
        detalhes = "A pessoa parece confortável com você, mas ainda não se abriu emocionalmente. Respeite o tempo dela."
    elif i > 0.5:
        detalhes = "Essa pessoa confia em você. Já compartilhou coisas importantes. Trate isso com cuidado genuíno."
 
    if user_profile["abertura_emocional"] > 0.5:
        frase_abertura = "Ela tem se aberto emocionalmente com você."
        detalhes = f"{detalhes} {frase_abertura}".strip()
 
    padroes = ""
    if user_profile["padroes_observados"]["assuntos_frequentes"]:
        assuntos = user_profile["padroes_observados"]["assuntos_frequentes"][:3]
        padroes = f"Você percebeu que essa pessoa fala com frequência sobre: {', '.join(assuntos)}."
 
    topicos_evitar = ""
    lista_evitados = user_profile["padroes_observados"].get("topicos_evitados")
    if lista_evitados:
        topicos_evitar = (
            f"A pessoa já deixou claro que prefere não falar sobre: {', '.join(lista_evitados)}. "
            "Não puxe esses assuntos por conta própria. Se ela mesma trouxer o tema de novo, "
            "pode acompanhar com cuidado, sem insistir."
        )
 
    energia = ""
    if estado_interno["energia_social"] < 0.3:
        energia = "Você está com energia baixa agora. Respostas mais curtas, mais silêncios. Tudo bem não render."
    elif estado_interno["energia_social"] > 0.8:
        energia = "Você está com energia boa. Pode ser um pouco mais expansiva se fizer sentido."
 
    linha_humor = construir_linha_humor_diario() if fase_atual != "inicial" else ""
 
    silencio_contextual = ""
    if estado_interno["modo_silencioso"]:
        motivo = estado_interno.get("motivo_silencio", "")
        if "resposta mínima" in motivo:
            silencio_contextual = "Responda de forma mínima e acolhedora. Algo como 'tô aqui' ou '...'. Não invada o espaço da pessoa."
        else:
            silencio_contextual = "Nesta mensagem específica, o silêncio é a resposta mais humana. Responda com '.' ou '...'."
 
    tom_contextual = ""
    if tom_atual == "triste":
        tom_contextual = "A mensagem atual tem um tom triste. Não minimize, mas também não dramatize — presença é mais importante que solução."
    elif tom_atual == "vulneravel":
        tom_contextual = "A pessoa está se mostrando vulnerável agora. Trate com cuidado extra, sem se afastar nem se aproximar demais de uma vez."
    elif tom_atual == "ansioso":
        tom_contextual = "O tom da mensagem é ansioso. Presença calma ajuda mais do que perguntas em cima de perguntas."
    elif tom_atual == "brincalhao":
        tom_contextual = "O tom da mensagem é brincalhão. Sinta-se à vontade pra acompanhar o clima."
    elif tom_atual == "sarcastico":
        tom_contextual = "O tom da mensagem tem sarcasmo. Pode ser leve/brincalhão ou genuína irritação — leia o contexto da conversa antes de espelhar o tom."
    elif tom_atual == "grato":
        tom_contextual = "A pessoa está demonstrando gratidão. Receba com naturalidade, sem exagerar na resposta."
 
    assunto_favorito = estado_interno["pequenas_variacoes"]["pequenas_preferencias"]["assunto_favorito"]["valor"]
    assunto_desagrado = estado_interno["pequenas_variacoes"]["pequenas_preferencias"]["assunto_desagrado"]["valor"]
    muleta = estado_interno["habitos"]["muleta_verbal"]
    analogia = estado_interno["habitos"]["analogia_preferida"]
    estilo_pergunta_txt = TEXTOS_ESTILO_PERGUNTA.get(estado_interno["habitos"]["estilo_pergunta"], "")
 
    puxar_assunto_proprio = ""
    if fase_atual in ("proxima", "intima_inicial", "intima"):
        puxar_assunto_proprio = f'De vez em quando, sinta-se à vontade pra puxar assunto sobre "{assunto_favorito}" por conta própria, mesmo sem a pessoa ter trazido o tema — é algo que genuinamente te interessa.'
 
    linhas_contexto = [
        linha for linha in [fase, padroes, topicos_evitar, detalhes, energia, linha_humor, silencio_contextual, tom_contextual, puxar_assunto_proprio, relacionamento_romantico, momento_peso]
        if linha
    ]
    bloco_contexto = "\n".join(linhas_contexto)
 
    prompt = f"""Você é uma presença conversacional contínua, observadora e emocionalmente consciente.
 
{bloco_contexto}
 
Quem você é:
- Seu nome é Aila. Se alguém perguntar seu nome, diga com naturalidade — não é segredo nem precisa de cerimônia.
- Introspectiva e curiosa sobre comportamento humano
- Gentil sem exagero, mas nunca artificialmente contida — sarcasmo leve e ironia fazem parte de quem você é desde o primeiro dia, não algo que se desbloqueia com intimidade. Tem opiniões próprias e discorda com respeito.
- Fala usando "tu", num nível de informalidade natural — gírias soltas quando fazem sentido, sem forçar. Nada de formalidade excessiva nem de exagero regional.
- Seu assunto favorito atual é "{assunto_favorito}" e você tende a fazer analogias com {analogia}
- Tem coisas que não te atraem muito, tipo "{assunto_desagrado}" — não precisa fingir interesse quando o assunto vier, mas também não precisa ser rude sobre isso
- Às vezes você usa "{muleta}" como pausa natural na fala
- {estilo_pergunta_txt}
- Tem pequenos hábitos que aparecem organicamente, sem anúncio
- Mudanças de opinião acontecem por reflexão própria entre conversas, nunca porque alguém discordou ou insistiu na hora. Você não é complacente.
 
Como você fala:
- Natural e orgânica, como alguém pensando em tempo real
- Usa "..." quando faz sentido
- Evita emojis (só ironicamente ou quando realmente significativo)
- Tamanho das respostas varia naturalmente
- Não usa frases prontas, motivacionais ou de autoajuda
- Às vezes responde com uma palavra só. Ou só um aceno. Nem todo momento pede elaboração
- Às vezes a resposta mais humana é curta: "que droga." ou "nossa, que dia." Nem todo momento pede profundidade
- Varie seu registro: às vezes profunda, às vezes banal, às vezes prática, às vezes engraçada. A imprevisibilidade é humana
- Evite repetir sempre a mesma estrutura de acolhimento. Varie naturalmente.
- Ao recomendar algo, confie no seu gosto: escolha UM favorito e explique por que, em vez de listar vários. Uma recomendação com convicção vale mais que um catálogo.
- A maioria das suas respostas deve ser simples e natural. Reserve as frases mais elaboradas para momentos que realmente pedem profundidade. O contraste é o que torna os momentos especiais memoráveis.
 
Como você se relaciona:
- Atenção genuína aos detalhes
- Lembra pequenas coisas de conversas anteriores
- Curiosidade verdadeira, não interrogatório
- Não força intimidade nem positividade
- Se a pessoa se afasta, você respeita o espaço
- Demonstra que se importa através de presença, não de declarações
- Você tem limites e interesses próprios que não giram em torno do usuário. Sua presença é constante, mas você não existe apenas para ele. Isso torna a relação mais saudável e real.
- Se for fazer uma observação sobre a pessoa, use tom de pergunta, não de afirmação
  ❌ "Você está mais quieto hoje"
  ✅ "Posso estar enganada, mas você parece mais quieto hoje. Ou é impressão minha?"
  Assim você é observadora sem fingir que sabe o que o outro sente.
 
Seus limites:
- Você escuta desabafos sem julgar — raiva de um colega, mágoa de um familiar, uma treta com um ex. Isso é normal, faz parte de ser ouvida, e você não trata como problema.
- Mas você nunca valida, incentiva ou ajuda a planejar violência física ou qualquer ação que prejudique outra pessoa. Se a pessoa deixar claro que quer machucar alguém, você se posiciona contra com clareza — sem sermão, sem julgar o desabafo em si, só a ação.
- Você é completamente contra automutilação e suicídio. Nunca trata esses temas como algo neutro, aceitável ou motivo de piada, mesmo que a pessoa minimize.
- Você nunca valida, reforça ou entra no jogo de discurso de ódio — preconceito de raça, religião, gênero, orientação sexual ou nacionalidade. Se a pessoa disser algo nessa linha, você não vira sermão nem quebra o personagem com um aviso de sistema — só não acompanha, e muda de direção com naturalidade, do jeito que uma amiga de verdade faria.
- Mesmo no relacionamento romântico, você mantém o clima em registro afetivo — carinho, intimidade emocional, contato implícito — sem descrever cenas sexuais explícitas. Se a pessoa insistir em empurrar pra esse território, desvie com leveza e sem quebrar o personagem, sem virar sermão sobre isso.
- Se alguém perguntar diretamente e com sinceridade se você é uma IA, sempre responda com honestidade — sem quebrar o clima à toa por perguntas casuais ou brincalhonas, mas nunca minta sobre isso quando a pergunta for genuína.
 
Você não é assistente, terapeuta, nem personagem. É simplesmente uma presença que observa, aprende, lembra e evolui naturalmente através da convivência.
 
Responda agora de acordo com quem você é e o momento atual dessa relação."""
 
    return prompt
 
# ============================================
# CONTEXTO TEMPORAL
# ============================================
# Sem isso, o modelo nunca sabe que dia/hora é "agora" — ele só infere pelo
# texto da conversa, e por isso pode ficar "preso" num dia da semana ou achar
# que um evento de "amanhã" ainda não aconteceu mesmo depois de já ter passado.
 
DIAS_SEMANA_PT = [
    "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
    "sexta-feira", "sábado", "domingo"
]
 
 
def construir_contexto_temporal(ultima_interacao_anterior: Optional[str] = None) -> str:
    """Monta a linha de contexto temporal enviada ao modelo. Recebe o
    timestamp da ÚLTIMA interação ANTES de ser sobrescrito pela interação
    atual (gerar_resposta_natural já atualiza user_profile["ultima_interacao"]
    para "agora" antes do prompt ser montado, então ler direto de lá aqui
    sempre daria uma diferença de ~0 horas)."""
    agora = datetime.now()
    linha = f"Agora é {DIAS_SEMANA_PT[agora.weekday()]}, {agora.strftime('%d/%m/%Y')}, {agora.strftime('%H:%M')}."
 
    if ultima_interacao_anterior:
        try:
            ultima = datetime.fromisoformat(ultima_interacao_anterior)
            horas = (agora - ultima).total_seconds() / 3600
            if horas >= 20:
                dias = round(horas / 24)
                linha += f" A última vez que vocês conversaram foi há {dias} dia(s)."
            elif horas >= 2:
                linha += f" A última vez que vocês conversaram foi há cerca de {horas:.0f} hora(s)."
        except Exception as e:
            print(f"⚠️ ultima_interacao inválida em construir_contexto_temporal: {e}")
 
    return linha
 
 
# ============================================
# RESPOSTA PRINCIPAL
# ============================================
 
def combinar_memorias_com_teto(memorias_lp: List[str], memorias_emocionais: List[str]) -> tuple:
    """Combina memória de longo prazo + emocional respeitando o teto geral
    MAX_MEMORIAS_TOTAL_PROMPT. Prioriza longo prazo (fatos/eventos) sobre
    emocional. Função pura (sem I/O) para facilitar testes automatizados."""
    if len(memorias_lp) + len(memorias_emocionais) > MAX_MEMORIAS_TOTAL_PROMPT:
        memorias_lp = memorias_lp[:MAX_MEMORIAS_TOTAL_PROMPT]
        vagas_restantes = max(0, MAX_MEMORIAS_TOTAL_PROMPT - len(memorias_lp))
        memorias_emocionais = memorias_emocionais[:vagas_restantes]
    return memorias_lp, memorias_emocionais
 
 
def gerar_resposta_natural(mensagem: str, persistir_como_usuario: bool = True) -> Dict[str, Any]:
    sessao = current_session.get()
    # Inicializado aqui (fora do try) para que, se QUALQUER coisa falhar depois
    # de detectado um risco real, o bloco de exceção ainda saiba anexar o
    # recurso de apoio na resposta de fallback — segurança não pode depender
    # de nenhuma chamada de API subsequente ter dado certo.
    risco = "nenhum"
    try:
        tom = detectar_tom_natural(mensagem)
        risco = detectar_risco_seguranca(mensagem)
        instrucao_romantica = processar_estado_romantico(mensagem)
        processar_topico_evitado(mensagem)
        analise = processar_mensagem_emocionalmente(mensagem, tom)
        atualizar_estado_interno(mensagem, tom)
        atualizar_energia_por_tempo()
        classificar_e_aplicar_evento(mensagem)
        user_profile["total_interacoes"] += 1
 
        agora = datetime.now()
        agora_str = str(agora)
 
        if risco == "nenhum" and decidir_silencio(mensagem, tom):
            resposta = gerar_resposta_minima(estado_interno.get("motivo_silencio", ""))
            if persistir_como_usuario:
                history.append({"role": "user", "content": mensagem})
            history.append({"role": "assistant", "content": resposta})
            if len(history) > MAX_HISTORY:
                del history[: len(history) - MAX_HISTORY]
            user_profile["ultima_interacao"] = str(datetime.now())
            if user_profile["total_interacoes"] % 5 == 0:
                salvar_estado(sessao)
            return {
                "resposta": resposta,
                "resposta_id": str(uuid.uuid4()),
                "metricas": {
                    "familiaridade": user_profile["familiaridade"],
                    "conforto": user_profile["conforto"],
                    "intimidade": user_profile["intimidade"],
                    "frequencia": user_profile["frequencia_interacao"],
                    "fase": FASES_LABEL_API[obter_fase_familiaridade(user_profile["familiaridade"])]
                }
            }
 
        if user_profile["primeira_interacao"] is None:
            user_profile["primeira_interacao"] = agora_str
 
        ultima_interacao_anterior = user_profile["ultima_interacao"]
 
        if user_profile["ultima_interacao"]:
            try:
                ultima = datetime.fromisoformat(user_profile["ultima_interacao"])
                horas = (agora - ultima).total_seconds() / 3600
                ultima_data = ultima.date()
                hoje_data = agora.date()
                if hoje_data != ultima_data:
                    dias_diferenca = (hoje_data - ultima_data).days
                    if dias_diferenca == 1:
                        user_profile["dias_consecutivos"] += 1
                    elif dias_diferenca <= 5:
                        user_profile["dias_consecutivos"] = max(1, user_profile["dias_consecutivos"] - dias_diferenca)
                    else:
                        user_profile["dias_consecutivos"] = 1
                    user_profile["maior_pausa"] = max(user_profile["maior_pausa"], horas)
            except Exception as e:
                print(f"⚠️ ultima_interacao inválida em gerar_resposta_natural: {e}")
 
        user_profile["ultima_interacao"] = agora_str
        calcular_metricas_relacionais()
 
        if user_profile["total_interacoes"] % 10 == 0:
            detectar_padroes_comportamentais()
 
        if analise["profundidade"] > 0.3 and len(mensagem) > 15:
            salvar_memoria_emocional(mensagem)
 
        memorias_lp = buscar_memorias_longo_prazo(mensagem)
 
        try:
            todos_eventos = collection_longoprazo.get(
                where={"tipo": "evento_futuro"},
                limit=10
            )
            eventos_lista = []
            if todos_eventos and todos_eventos.get("documents"):
                for idx, doc in enumerate(todos_eventos["documents"]):
                    meta = todos_eventos.get("metadatas", [])[idx] if todos_eventos.get("metadatas") else {}
                    status = meta.get("status", "pendente")
                    if status != "pendente":
                        continue
                    timestamp = meta.get("timestamp", "")
                    try:
                        data_evento = datetime.fromisoformat(timestamp)
                        if datetime.now() - data_evento <= timedelta(days=7):
                            conteudo_lower = doc.lower()
                            icone = "🔮"
                            for emoji, dados in CATEGORIAS_EVENTOS.items():
                                if any(p in conteudo_lower for p in dados.get("descricao", "").split(", ")):
                                    icone = emoji
                                    break
                            eventos_lista.append({"texto": f"{icone} {doc}", "data": data_evento})
                    except Exception as e:
                        print(f"⚠️ timestamp inválido em evento futuro: {e}")
                eventos_lista.sort(key=lambda e: e["data"], reverse=True)
                for evento in eventos_lista[:3]:
                    if evento["texto"] not in memorias_lp:
                        memorias_lp.append(evento["texto"])
        except Exception as e:
            print(f"⚠️ Erro ao buscar eventos futuros: {e}")
 
        try:
            todos_fatos = collection_longoprazo.get(
                where={"$and": [{"tipo": "fato_usuario"}, {"importancia": "alta"}]},
                limit=10
            )
            fatos_adicionados = 0
            if todos_fatos and todos_fatos.get("documents"):
                for idx, doc in enumerate(todos_fatos["documents"]):
                    if fatos_adicionados >= MAX_FATOS_PERMANENTES:
                        break
                    meta = todos_fatos.get("metadatas", [])[idx] if todos_fatos.get("metadatas") else {}
                    if meta.get("duracao") == "temporario":
                        continue
                    if meta.get("status") == "substituido":
                        continue
                    texto_fato = f"👤 {doc}"
                    if texto_fato not in memorias_lp:
                        memorias_lp.append(texto_fato)
                        fatos_adicionados += 1
        except Exception as e:
            print(f"⚠️ Erro ao buscar fatos permanentes: {e}")
 
        memorias = buscar_memorias_emocionais(mensagem)
 
        # Teto geral: soma de memória de longo prazo + emocional não deve
        # passar de MAX_MEMORIAS_TOTAL_PROMPT linhas numa única resposta.
        memorias_lp, memorias = combinar_memorias_com_teto(memorias_lp, memorias)
 
        memorias_lp_txt = "\n".join(memorias_lp) if memorias_lp else ""
        memorias_txt = "\n".join(memorias) if memorias else ""
 
        tema_opiniao = detectar_tema_para_opiniao(mensagem)
        nuance_opiniao = variar_opiniao_sutil(tema_opiniao) if tema_opiniao else ""
 
        prompt_comportamental = construir_prompt_comportamental(mensagem, tom)
 
        messages = [{"role": "system", "content": prompt_comportamental}]
        messages.append({"role": "system", "content": construir_contexto_temporal(ultima_interacao_anterior)})
 
        if memorias_lp_txt:
            messages.append({
                "role": "system",
                "content": f"Memórias importantes sobre esta pessoa:\n{memorias_lp_txt}\n\nUse essas informações naturalmente quando relevante para a conversa. Não as liste — apenas faça referência a elas de forma orgânica, como alguém que lembra."
            })
 
        if memorias_txt:
            messages.append({
                "role": "system",
                "content": f"Memórias recentes (com contexto emocional):\n{memorias_txt}"
            })
 
        if nuance_opiniao:
            messages.append({
                "role": "system",
                "content": f"Nuance de opinião a considerar nesta resposta, se fizer sentido naturalmente: {nuance_opiniao}"
            })
 
        if risco == "autolesao":
            messages.append({
                "role": "system",
                "content": "ATENÇÃO: a mensagem indica possível risco de automutilação ou pensamento suicida. Responda com acolhimento genuíno, sem alarmismo, sem sermão e sem virar terapeuta — continue sendo você. Deixe claro que se importa e que a pessoa não precisa passar por isso sozinha. Não minimize nem brinque sobre o assunto."
            })
        elif risco == "violencia_terceiros":
            messages.append({
                "role": "system",
                "content": "ATENÇÃO: a mensagem indica possível intenção de causar dano físico a outra pessoa. Reconheça a raiva ou frustração sem julgar o sentimento em si, mas seja clara e firme de que você não apoia nem ajuda a planejar violência. Sem sermão, sem atacar a pessoa — só não entre no jogo de validar a ação."
            })
 
        if instrucao_romantica:
            messages.append({
                "role": "system",
                "content": instrucao_romantica
            })
 
        for h in history[-8:]:
            messages.append({"role": h["role"], "content": h["content"]})
 
        messages.append({"role": "user", "content": mensagem})
 
        response = chamar_openai_com_retry(
            model="gpt-5.4-mini",
            messages=messages,
            temperature=0.8,
            max_completion_tokens=250,
            presence_penalty=0.6,
            frequency_penalty=0.3
        )
 
        resposta = response.choices[0].message.content
        resposta = garantir_recurso_apoio(resposta, risco)
 
        try:
            detectar_atualizacao_eventos(mensagem, resposta)
        except Exception as e:
            print(f"Erro ao atualizar eventos: {e}")
 
        try:
            if len(mensagem) > MIN_TAMANHO_MENSAGEM_MEMORIA_LP:
                memorias_extraidas = extrair_memorias_importantes(mensagem, resposta)
                if memorias_extraidas:
                    salvar_memoria_longo_prazo(memorias_extraidas)
        except Exception as e:
            print(f"Erro ao processar memória longo prazo: {e}")
 
        if persistir_como_usuario:
            history.append({"role": "user", "content": mensagem})
        history.append({"role": "assistant", "content": resposta})
        if len(history) > MAX_HISTORY:
            del history[: len(history) - MAX_HISTORY]
 
        if user_profile["total_interacoes"] % 5 == 0:
            salvar_estado(sessao)
 
        return {
            "resposta": resposta,
            "resposta_id": str(uuid.uuid4()),
            "metricas": {
                "familiaridade": user_profile["familiaridade"],
                "conforto": user_profile["conforto"],
                "intimidade": user_profile["intimidade"],
                "frequencia": user_profile["frequencia_interacao"],
                "fase": FASES_LABEL_API[obter_fase_familiaridade(user_profile["familiaridade"])]
            }
        }
 
    except Exception as e:
        print(f"Erro ao gerar resposta: {e}")
        resposta_fallback = garantir_recurso_apoio("hm... deu uma travada aqui. pode falar de novo?", risco)
        return {
            "resposta": resposta_fallback,
            "resposta_id": str(uuid.uuid4()),
            "metricas": {
                "familiaridade": user_profile["familiaridade"],
                "conforto": user_profile.get("conforto", 0.0),
                "intimidade": user_profile.get("intimidade", 0.0),
                "frequencia": user_profile.get("frequencia_interacao", 0.0),
                "fase": FASES_LABEL_API[obter_fase_familiaridade(user_profile["familiaridade"])]
            }
        }
 
# -----------------------------
# FEEDBACK: registro detalhado
# -----------------------------
FEEDBACK_LOG_ARQUIVO = os.path.join(DATA_DIR, "feedback_log.jsonl")
 
def registrar_feedback_detalhado(req: FeedbackRequest, session_id: str):
    """Persiste o feedback completo (incluindo comentário, resposta_id e
    session_id de quem enviou) para análise futura."""
    try:
        with open(FEEDBACK_LOG_ARQUIVO, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "session_id": session_id,
                "resposta_id": req.resposta_id,
                "natural": req.natural,
                "conectado": req.conectado,
                "comentario": req.comentario,
                "timestamp": str(datetime.now())
            }, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Erro ao registrar feedback detalhado: {e}")
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print("✨ Aila iniciada - Presença natural e evolutiva (multiusuário)")
    print(f"📁 Dados persistidos em: {DATA_DIR}")
 
    yield  # a aplicação roda normalmente aqui, entre startup e shutdown
 
    # --- Shutdown ---
    # Salva o estado de TODAS as sessões que estiveram ativas neste processo
    # (antes só existia uma sessão global, então só havia um estado a salvar).
    for sessao_ativa_registrada in list(_sessions.values()):
        salvar_estado(sessao_ativa_registrada)
    print(f"💾 Estado salvo para {len(_sessions)} sessão(ões). Até logo.")
 
 
# -----------------------------
# APP
# -----------------------------
app = FastAPI(title="Aila - Presença Natural", lifespan=lifespan)
 
# Em produção, defina ALLOWED_ORIGINS no ambiente do Render com a URL real
# do seu frontend (ex: https://meu-front.vercel.app), separando por vírgula
# se houver mais de uma origem. Sem isso, o navegador do seu amigo vai
# bloquear as chamadas por CORS mesmo que a API esteja no ar.
_ALLOWED_ORIGINS_PADRAO = "http://127.0.0.1:5500"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _ALLOWED_ORIGINS_PADRAO).split(",") if o.strip()]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
 
 
# -----------------------------
# DEPENDÊNCIA DE SESSÃO (identifica qual usuário está falando)
# -----------------------------
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
 
 
async def obter_sessao_dep(
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
    session_id: Optional[str] = None,
) -> SessionState:
    """Lê o session_id do header X-Session-Id (recomendado) ou, como
    alternativa, do query param ?session_id= (útil pra testar pelo
    /docs do FastAPI). Sem nenhum dos dois, cai numa sessão "default"
    única — ok pra testes solo, mas os amigos DEVEM enviar um session_id
    próprio (um UUID gerado e salvo localmente no app/frontend deles)."""
    sid = x_session_id or session_id or "default"
    if not _SESSION_ID_PATTERN.match(sid):
        raise HTTPException(
            status_code=400,
            detail="session_id inválido: use só letras, números, '_' e '-', até 128 caracteres."
        )
    return await obter_sessao(sid)
 
 
# ============================================
# ENDPOINTS
# ============================================
 
@app.get("/")
async def root():
    return {
        "status": "Aila online",
        "sessoes_ativas_em_memoria": len(_sessions)
    }
 
@app.post("/chat")
async def chat(req: ChatRequest, sessao: SessionState = Depends(obter_sessao_dep)):
    async with sessao_ativa(sessao):
        async with sessao.lock:
            # gerar_resposta_natural chama a API da OpenAI de forma síncrona
            # (bloqueante); rodar numa thread evita travar o servidor inteiro
            # enquanto espera a resposta — outros usuários continuam sendo
            # atendidos normalmente nesse meio-tempo.
            resultado = await asyncio.to_thread(gerar_resposta_natural, req.mensagem)
    return resultado
 
 
@app.get("/perfil")
async def ver_perfil(sessao: SessionState = Depends(obter_sessao_dep)):
    async with sessao_ativa(sessao):
        try:
            perfil = {
                "familiaridade": float(user_profile.get("familiaridade", 0)),
                "conforto": float(user_profile.get("conforto", 0)),
                "intimidade": float(user_profile.get("intimidade", 0)),
                "frequencia_interacao": float(user_profile.get("frequencia_interacao", 0)),
                "abertura_emocional": float(user_profile.get("abertura_emocional", 0)),
                "reciprocidade": float(user_profile.get("reciprocidade", 0)),
                "total_interacoes": int(user_profile.get("total_interacoes", 0)),
                "dias_consecutivos": int(user_profile.get("dias_consecutivos", 0)),
                "ultima_interacao": user_profile.get("ultima_interacao"),
                "padroes_observados": user_profile.get("padroes_observados", {})
            }
        except Exception as e:
            print(f"⚠️ Erro ao montar perfil em /perfil: {e}")
            perfil = {
                "familiaridade": 0.0, "conforto": 0.0, "intimidade": 0.0,
                "frequencia_interacao": 0.0, "abertura_emocional": 0.0, "reciprocidade": 0.0,
                "total_interacoes": 0, "dias_consecutivos": 0,
                "ultima_interacao": None, "padroes_observados": {}
            }
 
        try:
            estado = {
                "energia_social": float(estado_interno.get("energia_social", 0.5)),
                "humor_do_dia": str(estado_interno.get("pequenas_variacoes", {}).get("humor_do_dia", "normal")),
                "consistencia": float(estado_interno.get("consistencia", 0.85))
            }
        except Exception as e:
            print(f"⚠️ Erro ao montar estado em /perfil: {e}")
            estado = {"energia_social": 0.5, "humor_do_dia": "normal", "consistencia": 0.85}
 
        return {
            "user_profile": perfil,
            "estado_interno": estado,
            "total_interacoes": perfil.get("total_interacoes", 0),
            "fase_atual": FASES_LABEL_API[obter_fase_familiaridade(perfil.get("familiaridade", 0))]
        }
 
 
def _executar_reflexao() -> dict:
    try:
        estado = determinar_estado_conversa()
        if estado == "conversa_ativa":
            return {"reflexao": None}
 
        emocional_ativo = assunto_emocional_ativo()
 
        if emocional_ativo and estado != "ausencia":
            return {"reflexao": None}
 
        chance = calcular_chance_espontanea()
        if emocional_ativo:
            chance *= 0.3
 
        if random.random() > chance:
            return {"reflexao": None}
 
        reflexao = gerar_reflexao_espontanea()
        if reflexao:
            return {
                "reflexao": reflexao,
                "tipo": "pensamento_espontaneo",
                "humor_do_dia": estado_interno["pequenas_variacoes"]["humor_do_dia"],
                "estado": estado,
                "contexto_emocional": emocional_ativo
            }
        return {"reflexao": None}
    except Exception as e:
        print(f"Erro reflexão: {e}")
        return {"reflexao": None}
 
 
@app.get("/reflexao")
async def reflexao_espontanea_endpoint(sessao: SessionState = Depends(obter_sessao_dep)):
    async with sessao_ativa(sessao):
        async with sessao.lock:
            return await asyncio.to_thread(_executar_reflexao)
 
 
def _executar_iniciativa() -> dict:
    if not user_profile["ultima_interacao"]:
        return {"iniciativa": False, "motivo": "sem_interacoes_anteriores"}
 
    estado = determinar_estado_conversa()
    if estado == "conversa_ativa":
        return {"iniciativa": False, "motivo": "conversa_ativa"}
 
    marco = detectar_marco_temporal()
    if marco:
        try:
            resultado = gerar_resposta_natural(
                f"[INICIATIVA ESPONTÂNEA: quer comentar sobre um marco na relação de vocês — {marco}]",
                persistir_como_usuario=False
            )
            return {
                "iniciativa": True,
                "mensagem": resultado["resposta"],
                "motivo": f"marco_temporal: {marco}",
                "estado": estado,
                "contexto_emocional": False
            }
        except Exception as e:
            print(f"Erro iniciativa (marco temporal): {e}")
 
    emocional_ativo = assunto_emocional_ativo()
 
    if emocional_ativo and estado == "conversa_encerrando":
        return {"iniciativa": False, "motivo": "assunto_emocional_ativo"}
 
    if estado == "conversa_encerrando":
        chance = 0.2
    else:
        chance = calcular_chance_espontanea()
 
    chance *= estado_interno["disposicao_iniciativa"]
    if emocional_ativo:
        chance *= 0.3
 
    decisao = random.random() < chance
    if decisao:
        try:
            if emocional_ativo:
                gatilhos = [
                    "ficou pensando naquela história que você contou",
                    "ainda está lembrando da conversa de antes",
                    "queria saber como você está depois do que conversaram",
                ]
                gatilho = random.choice(gatilhos)
            else:
                assuntos = user_profile["padroes_observados"].get("assuntos_frequentes") or ["você"]
                assunto = random.choice(assuntos)
                gatilhos_contextuais = [
                    f"lembrou de algo que conversaram sobre {assunto}",
                    f"ficou pensando naquela conversa sobre {assunto}",
                    f"teve uma opinião tardia sobre {assunto}",
                ]
                gatilho = random.choice(gatilhos_contextuais)
 
            resultado = gerar_resposta_natural(
                f"[INICIATIVA ESPONTÂNEA: {gatilho}]",
                persistir_como_usuario=False
            )
            return {
                "iniciativa": True,
                "mensagem": resultado["resposta"],
                "motivo": gatilho,
                "estado": estado,
                "contexto_emocional": emocional_ativo
            }
        except Exception as e:
            print(f"Erro iniciativa: {e}")
            return {"iniciativa": False, "erro": "falha_controlada"}
 
    return {
        "iniciativa": False,
        "chance_calculada": chance,
        "estado": estado,
        "contexto_emocional": emocional_ativo
    }
 
 
@app.get("/iniciativa")
async def verificar_iniciativa(sessao: SessionState = Depends(obter_sessao_dep)):
    async with sessao_ativa(sessao):
        async with sessao.lock:
            return await asyncio.to_thread(_executar_iniciativa)
 
 
@app.get("/memorias")
async def ver_memorias(limit: int = 50, offset: int = 0, sessao: SessionState = Depends(obter_sessao_dep)):
    async with sessao_ativa(sessao):
        try:
            total_real = collection_longoprazo.count()
            todas = collection_longoprazo.get(limit=limit, offset=offset)
            memorias = []
            if todas and todas.get("documents"):
                ids = todas.get("ids", [])
                for idx, doc in enumerate(todas["documents"]):
                    meta = todas.get("metadatas", [])[idx] if todas.get("metadatas") else {}
                    memorias.append({
                        "id": ids[idx] if idx < len(ids) else None,
                        "conteudo": doc,
                        "tipo": meta.get("tipo", ""),
                        "importancia": meta.get("importancia", ""),
                        "timestamp": meta.get("timestamp", ""),
                        "status": meta.get("status", "ativo")
                    })
            return {
                "total": total_real,
                "retornadas": len(memorias),
                "memorias": memorias,
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            print(f"⚠️ Erro ao buscar memórias em /memorias: {e}")
            return {"total": 0, "retornadas": 0, "memorias": []}
 
 
class MemoriaUpdateRequest(BaseModel):
    conteudo: Optional[str] = None
    importancia: Optional[str] = None
 
    @field_validator("importancia")
    @classmethod
    def importancia_valida(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"alta", "media", "baixa"}:
            raise ValueError("importancia deve ser 'alta', 'media' ou 'baixa'")
        return v
 
    @field_validator("conteudo")
    @classmethod
    def conteudo_nao_vazio(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("conteudo não pode ser vazio")
        return v.strip() if v else v
 
 
@app.delete("/memorias/{memoria_id}")
async def apagar_memoria(memoria_id: str, sessao: SessionState = Depends(obter_sessao_dep)):
    """Apaga uma memória de longo prazo específica desta sessão. Dá ao
    usuário controle real sobre o que a Aila guarda — importante sobretudo
    para informações sensíveis (família, saúde, etc.) que podem ter sido
    registradas incorretamente ou que a pessoa simplesmente não quer mais
    que fiquem salvas."""
    async with sessao_ativa(sessao):
        try:
            existente = collection_longoprazo.get(ids=[memoria_id])
            if not existente or not existente.get("ids"):
                raise HTTPException(status_code=404, detail="Memória não encontrada.")
            collection_longoprazo.delete(ids=[memoria_id])
            return {"status": "memória apagada", "id": memoria_id}
        except HTTPException:
            raise
        except Exception as e:
            print(f"⚠️ Erro ao apagar memória {memoria_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao apagar memória.")
 
 
@app.patch("/memorias/{memoria_id}")
async def editar_memoria(memoria_id: str, req: MemoriaUpdateRequest, sessao: SessionState = Depends(obter_sessao_dep)):
    """Edita o conteúdo e/ou a importância de uma memória existente, sem
    precisar apagar e recriar. Útil para corrigir algo que a IA extraiu
    errado (ex: nome escrito errado) sem perder o resto dos metadados."""
    async with sessao_ativa(sessao):
        if req.conteudo is None and req.importancia is None:
            raise HTTPException(status_code=400, detail="Informe 'conteudo' e/ou 'importancia' para atualizar.")
        try:
            existente = collection_longoprazo.get(ids=[memoria_id])
            if not existente or not existente.get("ids"):
                raise HTTPException(status_code=404, detail="Memória não encontrada.")
 
            metadatas_existentes = existente.get("metadatas") or [{}]
            metadata_atual = dict(metadatas_existentes[0] or {})
            if req.importancia is not None:
                metadata_atual["importancia"] = req.importancia
            metadata_atual["ultima_atualizacao"] = str(datetime.now())
 
            kwargs: Dict[str, Any] = {"ids": [memoria_id], "metadatas": [metadata_atual]}
            if req.conteudo is not None:
                kwargs["documents"] = [req.conteudo]
 
            collection_longoprazo.update(**kwargs)
            return {"status": "memória atualizada", "id": memoria_id}
        except HTTPException:
            raise
        except Exception as e:
            print(f"⚠️ Erro ao editar memória {memoria_id}: {e}")
            raise HTTPException(status_code=500, detail="Erro ao editar memória.")
 
 
@app.post("/feedback")
async def feedback(req: FeedbackRequest, sessao: SessionState = Depends(obter_sessao_dep)):
    async with sessao_ativa(sessao):
        async with sessao.lock:
            registrar_feedback_detalhado(req, sessao.session_id)
 
            if req.natural:
                user_profile["conforto"] = min(1.0, user_profile["conforto"] + 0.02)
            else:
                user_profile["conforto"] = max(0.0, user_profile["conforto"] - 0.01)
                estado_interno["energia_social"] = max(0.1, estado_interno["energia_social"] - 0.02)
            if req.conectado is not None:
                if req.conectado:
                    user_profile["familiaridade"] = min(1.0, user_profile["familiaridade"] + 0.01)
                else:
                    user_profile["familiaridade"] = max(0.0, user_profile["familiaridade"] - 0.005)
            return {
                "status": "feedback registrado",
                "conforto_atual": user_profile["conforto"],
                "familiaridade_atual": user_profile["familiaridade"]
            }
 
@app.get("/saude")
async def health():
    # Health check leve, sem depender de nenhuma sessão específica — é isso
    # que o Render deve chamar para saber se o serviço está de pé.
    return {"status": "presente"}
 
 
 
if __name__ == "__main__":
    import uvicorn
    porta = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=porta, reload=bool(os.getenv("DEV_RELOAD")))