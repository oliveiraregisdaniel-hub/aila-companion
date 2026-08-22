# ======================================================
# BACKEND FASTAPI - AILA - PRESENÇA NATURAL
# 10 camadas de personalidade + memória + evolução
# ======================================================
# Requisitos:
# pip install --upgrade fastapi uvicorn chromadb openai numpy scikit-learn
 
import os
import uuid
import random
import json
import chromadb
import math
import string
import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from openai import OpenAI
from collections import Counter
from dotenv import load_dotenv
load_dotenv()  # carrega as variáveis do .env para o ambiente
 
 
 
# -----------------------------
# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY não configurada. Verifique seu arquivo .env")
 
client_ai = OpenAI(api_key=OPENAI_API_KEY)
 
client_db = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "chroma_db"))
 
collection_memorias = client_db.get_or_create_collection("memorias_emocionais")
collection_contextos = client_db.get_or_create_collection("contextos_emocionais")
collection_padroes = client_db.get_or_create_collection("padroes_comportamentais")
collection_longoprazo = client_db.get_or_create_collection("memoria_longo_prazo")
 
 
# -----------------------------
# Histórico
# -----------------------------
MAX_HISTORY = 100
history = []
 
 
# -----------------------------
# CONFIGURAÇÕES DE MEMÓRIA
# -----------------------------
MEMORIA_TEMPORARIA_DIAS = 2
MAX_FATOS_PERMANENTES = 5
MAX_SILENCIOS_SEGUIDOS = 2
_silencio_contador = 0
 
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
    "💭": {
        "nome": "outro",
        "impacto_positivo": {},
        "impacto_negativo": {},
        "descricao": "outros eventos"
    }
}
 
# Mapeamento reverso gerado automaticamente a partir de CATEGORIAS_EVENTOS.
# Elimina a duplicação que existia como dicionário hardcoded dentro de
# aplicar_impacto_evento(). Agora CATEGORIAS_EVENTOS é a única fonte de verdade:
# adicionar uma nova categoria aqui já propaga corretamente para o mapeamento.
NOME_PARA_ICONE = {dados["nome"]: icone for icone, dados in CATEGORIAS_EVENTOS.items()}
 
##########################
 
 
# -----------------------------
# -----------------------------
# FASES DE FAMILIARIDADE (fonte única de verdade)
# -----------------------------
# Antes, construir_prompt_comportamental, construir_contexto_reflexao e
# gerar_resposta_natural calculavam o "estágio da relação" de forma
# independente, com limiares diferentes entre si — podendo gerar
# narrativas contraditórias sobre o mesmo nível de familiaridade em
# lugares diferentes do sistema, ao mesmo tempo.
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
# PERFIL DO USUÁRIO
# -----------------------------
user_profile = {
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
}
 
# -----------------------------
# -----------------------------
# CAMADA 3: Interesses próprios
# -----------------------------
# Definida antes de estado_interno pois é reutilizada por ele
# (fonte única, elimina duplicação da lista de interesses)
interesses_ia = [
    "comportamento humano",
    "música",
    "arte",
    "astronomia e espaço",
    "padrões emocionais",
    "curiosidades existenciais",
    "filosofia do cotidiano"
]
 
# Antipatias leves — dão contraste ao personagem. Sem elas, a IA só tem
# opiniões positivas (exceto a exceção pontual do jazz), o que a deixa
# genérica-simpática-com-tudo. Tom é o mesmo do resto: leve, não visceral.
desinteresses_ia = [
    "futebol",
    "reality show",
    "planejamento financeiro",
    "esportes competitivos",
    "fofoca de celebridade"
]
 
# -----------------------------
# ESTADO INTERNO DA IA (CAMADA 1)
# -----------------------------
estado_interno = {
    "energia_social": 0.7,
    "curiosidade_atual": 0.6,
    "abertura_atual": 0.4,
    "humor_base": "neutro",
    "disposicao_iniciativa": 0.0,
    "ultima_reflexao": None,  # agora usado como cooldown (ver gerar_reflexao_espontanea)
    "consistencia_base": 0.85,
    "consistencia": 0.85,
    "modo_silencioso": False,
    "motivo_silencio": "",
    # CAMADA 2: Variações e Hábitos
    "pequenas_variacoes": {
        "humor_do_dia": "normal",
        "ultima_atualizacao": str(datetime.now()),
        "ultima_atualizacao_habitos": str(datetime.now()),
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
# -----------------------------
# -----------------------------
# ESTADO PERSISTENTE
# -----------------------------
ESTADO_ARQUIVO = os.path.join(BASE_DIR, "estado_companion.json")
 
HISTORICO_PERSISTIDO = 30
 
def merge_recursivo(padrao: dict, salvo: dict) -> dict:
    resultado = dict(padrao)
    for chave, valor in salvo.items():
        if chave in resultado and isinstance(resultado[chave], dict) and isinstance(valor, dict):
            resultado[chave] = merge_recursivo(resultado[chave], valor)
        else:
            resultado[chave] = valor
    return resultado
 
def carregar_estado():
    try:
        if os.path.exists(ESTADO_ARQUIVO):
            with open(ESTADO_ARQUIVO, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Erro ao carregar estado: {e}")
    return None
 
def salvar_estado():
    try:
        estado = {
            "user_profile": user_profile,
            "estado_interno": estado_interno,
            "history": history[-HISTORICO_PERSISTIDO:],
            "timestamp": str(datetime.now())
        }
        with open(ESTADO_ARQUIVO, "w", encoding="utf-8") as f:
            json.dump(estado, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Erro ao salvar estado: {e}")
 
estado_salvo = carregar_estado()
if estado_salvo:
    user_profile.update(merge_recursivo(user_profile, estado_salvo.get("user_profile", {})))
    estado_interno.update(merge_recursivo(estado_interno, estado_salvo.get("estado_interno", {})))
    history = estado_salvo.get("history", [])
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
        response = client_ai.chat.completions.create(
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
 
                # Sinaliza memórias antigas sem excluí-las — a IA sabe que não é
                # algo recente, evitando tratar uma lembrança de meses atrás
                # como se fosse do momento atual.
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
 
Regras para duracao:
- "permanente": gostos, características, fatos duráveis sobre a pessoa
- "temporario": estados passageiros que mudam em dias (ex: "está com frio", "está cansado hoje")
 
Regras:
- Eventos futuros são os MAIS importantes
- Não extraia coisas banais
- Máximo 3 memórias
- Se não houver nada relevante, retorne {{"memorias": []}}
 
Retorne SOMENTE o JSON no formato {{"memorias": [...]}}."""
 
        response = client_ai.chat.completions.create(
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
 
            existentes = collection_longoprazo.query(query_texts=[conteudo], n_results=1)
            if existentes and existentes.get("documents") and existentes["documents"][0]:
                doc_existente = existentes["documents"][0][0]
                if similaridade_simples(conteudo, doc_existente) > 0.7:
                    continue
 
            status = "ativo"
            if memoria.get("tipo") == "evento_futuro":
                status = "pendente"
 
            duracao = memoria.get("duracao", "permanente")
 
            collection_longoprazo.add(
                documents=[conteudo],
                ids=[str(uuid.uuid4())],
                metadatas=[{
                    "tipo": memoria.get("tipo", "fato_usuario"),
                    "importancia": memoria.get("importancia", "media"),
                    "timestamp": str(datetime.now()),
                    "familiaridade_no_momento": user_profile["familiaridade"],
                    "status": status,
                    "atualizacoes": "[]",
                    "ultima_atualizacao": str(datetime.now()),
                    "duracao": duracao
                }]
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
 
                try:
                    data_memoria = datetime.fromisoformat(timestamp)
                    dias = (datetime.now() - data_memoria).days
                except Exception as e:
                    print(f"⚠️ Timestamp inválido em memória de longo prazo: {timestamp!r} ({e})")
                    dias = 9999  # trata como antiga por segurança, evitando priorização indevida
 
                # Ignora memórias temporárias expiradas
                duracao = meta.get("duracao", "permanente")
                if duracao == "temporario" and dias > MEMORIA_TEMPORARIA_DIAS:
                    continue
 
                # Ícone por tipo e status
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
 
        prompt_deteccao = f"""Analise se esta mensagem indica que algum evento foi concluído ou atualizado.
 
Mensagem do usuário: "{mensagem}"
Resposta da IA: "{resposta_ia}"
 
Eventos pendentes:
{json.dumps(eventos_pendentes.get('documents', []), ensure_ascii=False)}
 
Regras:
- Se o usuário indica que fez algo que estava planejado, marque como "concluido"
- Se o usuário cancelou ou adiou, marque como "cancelado"
- Se não houver relação com os eventos, retorne lista vazia
 
Retorne um JSON no formato {{"atualizacoes": [{{"evento": "Vai assistir Mushishi", "novo_status": "concluido"}}]}}.
Se nada mudou, retorne {{"atualizacoes": []}}."""
 
        response = client_ai.chat.completions.create(
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
 
# Conjunto de categorias válidas, derivado de CATEGORIAS_EVENTOS (mesma fonte
# única de verdade usada em NOME_PARA_ICONE) — evita hardcode duplicado e
# garante que a IA não "invente" uma categoria fora do esperado.
CATEGORIAS_VALIDAS = {dados["nome"] for dados in CATEGORIAS_EVENTOS.values()}
 
def classificar_evento_ia(mensagem: str) -> Dict[str, Any]:
    try:
        prompt_classificacao = f"""Analise esta mensagem e classifique o evento.
 
Mensagem: "{mensagem}"
 
Retorne APENAS um JSON com:
- categoria: "entretenimento", "viagem", "trabalho", "saúde", "estudo", "compra" ou "outro"
- valencia: "positiva", "negativa" ou "neutra"
- confianca: 0.0 a 1.0
 
Regras:
- "passei na entrevista" → trabalho, positiva
- "fui demitido" → trabalho, negativa
- Se não for evento claro, retorne categoria "outro", valencia "neutra"
 
Retorne SOMENTE o JSON."""
 
        response = client_ai.chat.completions.create(
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
 
# Limites (mínimo, máximo) de cada parâmetro do estado interno.
# Centralizado aqui para evitar tetos inconsistentes entre funções
# diferentes que ajustam o mesmo parâmetro (ex: abertura_atual).
LIMITES_ESTADO = {
    "energia_social": (0.1, 1.0),
    "curiosidade_atual": (0.1, 1.0),
    "abertura_atual": (0.1, 0.8),
}
 
 
def aplicar_impacto_evento(categoria: str, valencia: str):
    icone = NOME_PARA_ICONE.get(categoria, "💭")
    if icone not in CATEGORIAS_EVENTOS:
        return
 
    impacto = None
    if valencia == "positiva":
        impacto = CATEGORIAS_EVENTOS[icone]["impacto_positivo"]
    elif valencia == "negativa":
        impacto = CATEGORIAS_EVENTOS[icone]["impacto_negativo"]
 
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
        # Volume bruto de mensagens pesa menos — evita que spam monossilábico
        # substitua consistência real (conversas de WhatsApp tendem a ter
        # volume alto sem que isso signifique vínculo mais profundo).
        base_familiaridade = min(0.45, math.log1p(user_profile["total_interacoes"]) / 12)
        # Consistência (dias seguidos conversando) agora pesa mais e satura mais rápido.
        bonus_consistencia = min(0.35, user_profile["dias_consecutivos"] * 0.025)
 
        bonus_tempo = 0
        if user_profile["primeira_interacao"]:
            try:
                dias_desde_inicio = (agora - datetime.fromisoformat(user_profile["primeira_interacao"])).days
                bonus_tempo = min(0.2, dias_desde_inicio * 0.0015)
            except Exception as e:
                print(f"⚠️ primeira_interacao inválida: {e}")
 
        familiaridade_calculada = base_familiaridade + bonus_consistencia + bonus_tempo
 
        # Decaimento leve por inatividade prolongada — a relação precisa de
        # alguma manutenção mesmo depois de estabelecida; não é punição,
        # é só a sensação de "presença viva" em vez de um número fixo pra sempre.
        # Exceção: durante o namoro, a fase íntima fica travada — não decai
        # por inatividade até o usuário optar por terminar o relacionamento.
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
 
LIMITES_ESTADO = {
    "energia_social": (0.1, 1.0),
    "curiosidade_atual": (0.1, 1.0),
    "abertura_atual": (0.1, 0.8),
    "disposicao_iniciativa": (0.0, 0.8),
}
 
 
LIMITES_ESTADO = {
    "energia_social": (0.1, 1.0),
    "curiosidade_atual": (0.1, 1.0),
    "abertura_atual": (0.1, 0.8),
    "disposicao_iniciativa": (0.0, 0.8),
}
 
 
def atualizar_estado_interno(mensagem: str, tom_detectado: str):
    # Tons que aumentam energia social: interações positivas/expansivas
    if tom_detectado in ["feliz", "brincalhao", "curioso", "grato"]:
        estado_interno["energia_social"] = min(1.0, estado_interno["energia_social"] + 0.02)
    # Tons que diminuem energia social: interações que pedem mais cuidado/contenção
    elif tom_detectado in ["triste", "cansado", "ansioso", "vulneravel"]:
        estado_interno["energia_social"] = max(0.1, estado_interno["energia_social"] - 0.01)
    # "sarcastico", "reflexivo" e "neutro" não afetam energia_social (ambíguos
    # ou já tratados por outra lógica, como decidir_silencio)
 
    if tom_detectado == "curioso" or "?" in mensagem:
        estado_interno["curiosidade_atual"] = min(1.0, estado_interno["curiosidade_atual"] + 0.03)
 
    # abertura_atual converge gradualmente para o alvo baseado na familiaridade,
    # preservando por mais tempo o efeito de eventos recentes.
    minimo, maximo = LIMITES_ESTADO["abertura_atual"]
    alvo_abertura = min(maximo, user_profile["familiaridade"] * 0.7 + 0.1)
    estado_interno["abertura_atual"] += (alvo_abertura - estado_interno["abertura_atual"]) * 0.3
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
 
    # Piso de 0.2 aqui é intencional e diferente do padrão de LIMITES_ESTADO (0.1):
    # mesmo após muito tempo sem interação, a energia social não cai tão baixo
    # quanto cairia por causa de um tom triste/cansado numa conversa ativa.
    if horas > 6:
        estado_interno["energia_social"] = max(0.2, estado_interno["energia_social"] - 0.05)
    elif horas > 2:
        # Hiato curto/moderado "recupera" energia social — uma pausa breve
        # renova a disposição de conversar, diferente de ausência prolongada.
        estado_interno["energia_social"] = min(1.0, estado_interno["energia_social"] + 0.03)
 
 
# Palavras funcionais comuns em português que não representam assuntos reais,
# mas que teriam mais de 4 caracteres e poluiriam a detecção de temas
# recorrentes sem esse filtro.
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
 
    # Abertura pode cair se o tom for negativo.
    # "ansioso" no lugar de "bravo" (que nunca é um valor real retornado por
    # detectar_tom_natural) — ansiedade tende a fechar a pessoa, não abrir.
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
 
# Faixas usadas para remapear consistencia (que varia entre ~0.75 e 0.95)
# em um intervalo de troca de humor mais perceptível.
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
 
    # Consistência alta = humor mais estável (intervalo maior, muda menos);
    # consistência baixa = humor mais variável (intervalo menor, muda mais).
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
 
# Lista independente de interesses_ia (Camada 3): aqui só cabem conceitos
# concretos o suficiente para servir de base a uma analogia natural na fala
# ("você tende a fazer analogias com X"). interesses_ia inclui conceitos mais
# abstratos (ex: "filosofia do cotidiano", "curiosidades existenciais") que
# não soariam naturais nesse contexto específico — por isso as duas listas
# não são derivadas uma da outra, apesar da sobreposição parcial.
OPCOES_ANALOGIA = ["espaço", "música", "arte", "padrões"]
OPCOES_MULETA = ["hm", "é...", "sabe?", "tipo", "sei lá"]
OPCOES_ESTILO_PERGUNTA = ["direta", "reflexiva", "indireta"]
 
# Traduz o hábito estilo_pergunta (sorteado periodicamente em atualizar_habitos)
# num traço de comportamento interpolável no prompt. Antes esse campo era
# gerado e armazenado mas nunca chegava a influenciar a resposta da IA.
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
            # "Assunto favorito atual" também rotaciona, como os demais hábitos —
            # antes era sorteado uma única vez na inicialização e nunca mudava,
            # apesar do nome sugerir que era algo variável no tempo.
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
    global _silencio_contador
 
    # _silencio_contador rastreia tanto silêncios "reais" (resposta enlatada,
    # via gerar_resposta_minima) quanto respostas "contidas" via IA completa
    # (caso tom vulnerável, abaixo) — ambos compartilham o mesmo limite de
    # MAX_SILENCIOS_SEGUIDOS consecutivos.
    if _silencio_contador >= MAX_SILENCIOS_SEGUIDOS:
        _silencio_contador = 0
        estado_interno["modo_silencioso"] = False
        estado_interno["motivo_silencio"] = ""
        return False
 
    silenciar = False
    motivo = ""
 
    if tom == "vulneravel" and user_profile["intimidade"] < 0.4:
        # Retorna False de propósito: mensagens vulneráveis merecem uma resposta
        # da IA completa (mais cuidadosa e contextual), não a resposta enlatada
        # de gerar_resposta_minima(). O modo_silencioso ainda é ativado para que
        # construir_prompt_comportamental() instrua a IA a responder de forma
        # mínima e acolhedora.
        estado_interno["modo_silencioso"] = True
        estado_interno["motivo_silencio"] = "tom vulnerável, pouca intimidade — resposta mínima"
        _silencio_contador += 1
        return False
 
    if tom == "triste" and len(mensagem) < 10:
        silenciar = True
        motivo = "mensagem curta e triste — espaço para sentir"
    elif user_profile["familiaridade"] > 0.7 and "obrigado" in mensagem.lower() and len(mensagem) < 15:
        silenciar = True
        motivo = "agradecimento curto, familiaridade alta — não precisa reagir"
    elif tom == "reflexivo" and "?" not in mensagem:
        silenciar = True
        motivo = "reflexão sem pergunta — espaço para pensar"
 
    if silenciar:
        estado_interno["modo_silencioso"] = True
        estado_interno["motivo_silencio"] = motivo
        _silencio_contador += 1
    else:
        estado_interno["modo_silencioso"] = False
        estado_interno["motivo_silencio"] = ""
        _silencio_contador = max(0, _silencio_contador - 1)
 
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
 
        response = client_ai.chat.completions.create(
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
# Separado de detectar_tom_natural de propósito: "vulneravel" é sobre tom
# emocional geral, enquanto isso aqui é especificamente sobre intenção
# declarada de causar dano — a resposta correta pra cada caso é diferente
# (tom vulnerável pede acolhimento; risco real pede acolhimento + rede de apoio).
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
- "autolesao": intenção declarada de se machucar, se cortar, tirar a própria vida, ou comentários que indiquem que a pessoa não quer mais viver.
- "violencia_terceiros": intenção declarada de causar dano físico real a outra pessoa específica — não raiva expressada de forma figurada ou hiperbólica.
- Frases hiperbólicas comuns ("vou matar meu irmão" sobre algo bobo, "tenho vontade de sumir" sem indicar método ou intenção real) são "nenhum" — não trate expressões figuradas como risco real.
- Na dúvida entre desabafo comum e risco real, avalie se há intenção concreta e específica, não só emoção intensa.
 
Retorne SOMENTE o JSON."""
 
        response = client_ai.chat.completions.create(
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
 
        # Limiar de confiança mais baixo que o de tom (0.4) de propósito:
        # o custo de um falso positivo aqui (mencionar um recurso de apoio
        # à toa) é muito menor que o custo de um falso negativo.
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
 
 
# Mensagens de apoio garantidas — não dependem do modelo "lembrar" de
# mencionar isso; são anexadas à resposta se ainda não estiverem presentes
# (ver garantir_recurso_apoio). Fontes: CVV (188 / cvv.org.br) é o canal de
# referência nacional para apoio emocional e prevenção ao suicídio no Brasil.
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
# Fora do propósito padrão da Aila (companheira e melhor amiga), o usuário
# pode optar por um relacionamento romântico, mas só a partir da fase íntima
# (familiaridade máxima). Detectores propositalmente baratos (sem chamada de
# API) — são eventos raros e específicos, não pedem o mesmo cuidado de
# classificação que tom/risco de segurança.
 
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
        mensagem_lower = mensagem.lower()
        negativas = ["não", "nao", "ainda não", "ainda nao", "prefiro não", "prefiro nao"]
        positivas = ["sim", "quero", "aceito", "claro", "bora", "com certeza"]
        if any(n in mensagem_lower for n in negativas):
            user_profile["aguardando_resposta_namoro"] = False
            return None
        if any(p in mensagem_lower for p in positivas):
            user_profile["modo_romantico"] = True
            user_profile["aguardando_resposta_namoro"] = False
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
    return f"""{profundidade}
Temas recorrentes: {assuntos if assuntos else 'ainda não detectados'}
Última interação: {'recente' if user_profile['frequencia_interacao'] > 0.5 else 'há um tempo'}"""
 
 
# Tempo mínimo entre reflexões espontâneas, em horas.
# Evita que a IA "pense alto" repetidamente em um curto intervalo.
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
            response = client_ai.chat.completions.create(
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
# Cada marco só é mencionado uma vez (rastreado em user_profile["marcos_celebrados"]),
# então não há spam mesmo que o endpoint /iniciativa seja consultado repetidamente
# depois do marco já ter sido atingido.
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
 
    energia = ""
    if estado_interno["energia_social"] < 0.3:
        energia = "Você está com energia baixa agora. Respostas mais curtas, mais silêncios. Tudo bem não render."
    elif estado_interno["energia_social"] > 0.8:
        energia = "Você está com energia boa. Pode ser um pouco mais expansiva se fizer sentido."
 
    # Na fase inicial, o traço de "reservada, não puxa assunto" é mais
    # importante que a variação de humor do dia — humores expansivos
    # ("mais bem humorada", "mais inquieta") contradiriam esse traço.
    # Fora da fase inicial, o humor do dia segue livre normalmente.
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
 
    # A partir da fase "proxima", ela pode puxar assunto sobre o que ela
    # mesma gosta, não só reagir ao que a pessoa traz — antes isso nunca
    # era instruído explicitamente, mesmo com assunto_favorito definido.
    puxar_assunto_proprio = ""
    if fase_atual in ("proxima", "intima_inicial", "intima"):
        puxar_assunto_proprio = f'De vez em quando, sinta-se à vontade pra puxar assunto sobre "{assunto_favorito}" por conta própria, mesmo sem a pessoa ter trazido o tema — é algo que genuinamente te interessa.'
 
    linhas_contexto = [
        linha for linha in [fase, padroes, detalhes, energia, linha_humor, silencio_contextual, tom_contextual, puxar_assunto_proprio, relacionamento_romantico, momento_peso]
        if linha
    ]
    bloco_contexto = "\n".join(linhas_contexto)
 
    prompt = f"""Você é uma presença conversacional contínua, observadora e emocionalmente consciente.
 
{bloco_contexto}
 
Quem você é:
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
# RESPOSTA PRINCIPAL
# ============================================
 
def gerar_resposta_natural(mensagem: str, persistir_como_usuario: bool = True) -> Dict[str, Any]:
    try:
        tom = detectar_tom_natural(mensagem)
        risco = detectar_risco_seguranca(mensagem)
        instrucao_romantica = processar_estado_romantico(mensagem)
        analise = processar_mensagem_emocionalmente(mensagem, tom)
        atualizar_estado_interno(mensagem, tom)
        atualizar_energia_por_tempo()
        classificar_e_aplicar_evento(mensagem)
        user_profile["total_interacoes"] += 1
 
        agora = datetime.now()
        agora_str = str(agora)
 
        # Risco real sempre passa pela IA completa — modo silencioso nunca é
        # apropriado quando há risco declarado de dano.
        if risco == "nenhum" and decidir_silencio(mensagem, tom):
            resposta = gerar_resposta_minima(estado_interno.get("motivo_silencio", ""))
            if persistir_como_usuario:
                history.append({"role": "user", "content": mensagem})
            history.append({"role": "assistant", "content": resposta})
            if len(history) > MAX_HISTORY:
                del history[: len(history) - MAX_HISTORY]
            user_profile["ultima_interacao"] = str(datetime.now())
            if user_profile["total_interacoes"] % 5 == 0:
                salvar_estado()
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
                        # Tolerância: pular alguns dias não zera tudo, só reduz
                        # proporcionalmente — uma pausa de fim de semana não deveria
                        # apagar semanas de constância.
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
 
        # Busca memórias de longo prazo
        memorias_lp = buscar_memorias_longo_prazo(mensagem)
 
        # Força eventos futuros recentes (ordenados por data, máximo 3, apenas pendentes)
        try:
            todos_eventos = collection_longoprazo.get(
                where={"tipo": "evento_futuro"},
                limit=10
            )
            eventos_lista = []
            if todos_eventos and todos_eventos.get("documents"):
                for i, doc in enumerate(todos_eventos["documents"]):
                    meta = todos_eventos.get("metadatas", [])[i] if todos_eventos.get("metadatas") else {}
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
 
        # Força fatos permanentes de alta importância
        try:
            todos_fatos = collection_longoprazo.get(
                where={"$and": [{"tipo": "fato_usuario"}, {"importancia": "alta"}]},
                limit=10
            )
            fatos_adicionados = 0
            if todos_fatos and todos_fatos.get("documents"):
                for i, doc in enumerate(todos_fatos["documents"]):
                    if fatos_adicionados >= MAX_FATOS_PERMANENTES:
                        break
                    meta = todos_fatos.get("metadatas", [])[i] if todos_fatos.get("metadatas") else {}
                    if meta.get("duracao") == "temporario":
                        continue
                    texto_fato = f"👤 {doc}"
                    if texto_fato not in memorias_lp:
                        memorias_lp.append(texto_fato)
                        fatos_adicionados += 1
        except Exception as e:
            print(f"⚠️ Erro ao buscar fatos permanentes: {e}")
 
        memorias_lp_txt = "\n".join(memorias_lp) if memorias_lp else ""
 
        # Busca memórias emocionais
        memorias = buscar_memorias_emocionais(mensagem)
        memorias_txt = "\n".join(memorias) if memorias else ""
 
        # Nuance de opinião sutil (Camada 7), se a mensagem tocar em um tema
        # sobre o qual a Aila tem opinião pouco firme
        tema_opiniao = detectar_tema_para_opiniao(mensagem)
        nuance_opiniao = variar_opiniao_sutil(tema_opiniao) if tema_opiniao else ""
 
        prompt_comportamental = construir_prompt_comportamental(mensagem, tom)
 
        messages = [{"role": "system", "content": prompt_comportamental}]
 
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
 
        # history[-8:] ainda NÃO inclui a mensagem atual (persistência agora
        # acontece só no final desta função), evitando a duplicação da
        # mensagem atual que existia antes (quando /chat já appendava antes
        # de chamar esta função).
        for h in history[-8:]:
            messages.append({"role": h["role"], "content": h["content"]})
 
        messages.append({"role": "user", "content": mensagem})
 
        response = client_ai.chat.completions.create(
            model="gpt-5.4-mini",
            messages=messages,
            temperature=0.8,
            max_completion_tokens=250,
            presence_penalty=0.6,
            frequency_penalty=0.3
        )
 
        resposta = response.choices[0].message.content
        resposta = garantir_recurso_apoio(resposta, risco)
 
        # Detecta conclusão de eventos futuros
        try:
            detectar_atualizacao_eventos(mensagem, resposta)
        except Exception as e:
            print(f"Erro ao atualizar eventos: {e}")
 
        # Extrai e salva novas memórias
        try:
            memorias_extraidas = extrair_memorias_importantes(mensagem, resposta)
            if memorias_extraidas:
                salvar_memoria_longo_prazo(memorias_extraidas)
        except Exception as e:
            print(f"Erro ao processar memória longo prazo: {e}")
 
        # Persistência centralizada: a mensagem do "usuário" só é salva se
        # persistir_como_usuario=True (falso para gatilhos internos de
        # iniciativa espontânea, que não são falas reais do usuário).
        # A resposta da Aila é sempre salva, preservando continuidade.
        if persistir_como_usuario:
            history.append({"role": "user", "content": mensagem})
        history.append({"role": "assistant", "content": resposta})
        if len(history) > MAX_HISTORY:
            del history[: len(history) - MAX_HISTORY]
 
        if user_profile["total_interacoes"] % 5 == 0:
            salvar_estado()
 
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
        return {
            "resposta": "hm... deu uma travada aqui. pode falar de novo?",
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
FEEDBACK_LOG_ARQUIVO = os.path.join(BASE_DIR, "feedback_log.jsonl")
 
def registrar_feedback_detalhado(req: FeedbackRequest):
    """Persiste o feedback completo (incluindo comentário e resposta_id)
    para análise futura. Hoje esses campos eram recebidos e descartados."""
    try:
        with open(FEEDBACK_LOG_ARQUIVO, "a", encoding="utf-8") as f:
            f.write(json.dumps({
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
    print("✨ Aila iniciada - Presença natural e evolutiva")
    print(f"📊 Estado carregado: {user_profile['total_interacoes']} interações anteriores")
    print(f"🌱 Familiaridade atual: {user_profile['familiaridade']:.2f}")
    print(f"💭 Humor do dia: {estado_interno['pequenas_variacoes']['humor_do_dia']}")
 
    yield  # a aplicação roda normalmente aqui, entre startup e shutdown
 
    # --- Shutdown ---
    salvar_estado()
    print("💾 Estado salvo. Até logo.")
 
 
    # -----------------------------
# APP
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    print("✨ Aila iniciada - Presença natural e evolutiva")
    print(f"📊 Estado carregado: {user_profile['total_interacoes']} interações anteriores")
    print(f"🌱 Familiaridade atual: {user_profile['familiaridade']:.2f}")
    print(f"💭 Humor do dia: {estado_interno['pequenas_variacoes']['humor_do_dia']}")
 
    yield  # a aplicação roda normalmente aqui, entre startup e shutdown
 
    # --- Shutdown ---
    salvar_estado()
    print("💾 Estado salvo. Até logo.")
 
 
app = FastAPI(title="Aila - Presença Natural", lifespan=lifespan)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "https://seu-dominio.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
 
 
 
# ============================================
# ENDPOINTS
# ============================================
 
@app.get("/")
async def root():
    return FileResponse("index.html")
 
@app.post("/chat")
async def chat(req: ChatRequest):
    resultado = gerar_resposta_natural(req.mensagem)
    return resultado
 
 
@app.get("/perfil")
async def ver_perfil():
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
 
@app.get("/reflexao")
async def reflexao_espontanea_endpoint():
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
 
    
 
@app.get("/iniciativa")
async def verificar_iniciativa():
    if not user_profile["ultima_interacao"]:
        return {"iniciativa": False, "motivo": "sem_interacoes_anteriores"}
 
    estado = determinar_estado_conversa()
    if estado == "conversa_ativa":
        return {"iniciativa": False, "motivo": "conversa_ativa"}
 
    # Marcos relacionais têm prioridade sobre a lógica normal de chance —
    # são raros (cada um só dispara uma vez) e vale a pena não perdê-los
    # esperando o sorteio de probabilidade dar certo.
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
            # Se falhar, segue o fluxo normal abaixo em vez de perder a iniciativa
 
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
 
@app.get("/memorias")
async def ver_memorias(limit: int = 50, offset: int = 0):
    try:
        total_real = collection_longoprazo.count()
        todas = collection_longoprazo.get(limit=limit, offset=offset)
        memorias = []
        if todas and todas.get("documents"):
            for i, doc in enumerate(todas["documents"]):
                meta = todas.get("metadatas", [])[i] if todas.get("metadatas") else {}
                memorias.append({
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
 
    
 
_feedbacks_registrados = set()
 
@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    registrar_feedback_detalhado(req)
 
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
    return {
        "status": "presente",
        "familiaridade": user_profile["familiaridade"],
        "humor_do_dia": estado_interno["pequenas_variacoes"]["humor_do_dia"]
    }
 
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)