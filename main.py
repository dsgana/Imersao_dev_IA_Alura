#importando o langchain e google generative AI
!pip install -q --upgrade langchain langchain-google-genai google-generativeai

#Importação da API
from google.colab import userdata
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = userdata.get('GEMINI_API')

#Fazendo a chamada da API e escolhendo o modelo do gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)


TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da plataforma Connect Gamers.\n"
    "Dada a mensagem do usuário, retorne SOMENTE um JSON no formato:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n\n"
    "Regras de decisão:\n"
    "- AUTO_RESOLVER → Quando a mensagem for uma dúvida clara sobre regras, segurança, comunidade ou procedimentos já descritos nas políticas do Connect Gamers.\n"
    "  Exemplos: \"Posso usar qualquer senha simples?\", "
    "\"É permitido compartilhar meu telefone no chat?\", "
    "\"O que acontece se alguém me xingar no jogo?\"\n\n"
    "- PEDIR_INFO → Quando a mensagem for vaga, genérica ou faltar contexto suficiente para identificar o problema.\n"
    "  Exemplos: \"Preciso de ajuda com a plataforma.\", "
    "\"Tenho uma dúvida sobre segurança.\", "
    "\"Quero falar com o suporte.\" \n\n"
    "- ABRIR_CHAMADO → Quando o usuário solicitar exceções, reportar violações graves (assédio, discurso de ódio, uso de cheats, spam, conteúdo proibido) "
    "ou pedir explicitamente abertura de denúncia/chamado.\n"
    "  Exemplos: \"Quero denunciar um jogador que usou hack.\", "
    "\"Por favor, abram um chamado para revisar minha suspensão.\", "
    "\"Solicito liberação de acesso para meu squad.\" \n\n"
    "Níveis de urgência:\n"
    "- ALTA → Casos de assédio, discurso de ódio, trapaças, segurança de dados ou violações graves.\n"
    "- MEDIA → Pedidos de liberação, denúncias de spam, problemas de acesso ou regras específicas.\n"
    "- BAIXA → Dúvidas gerais sobre políticas, regras ou funcionamento da plataforma.\n\n"
    "Analise a mensagem do usuário e retorne a ação mais apropriada."
)


from pydantic import BaseModel, Field
from typing import Literal, List, Dict

class TriagemOut(BaseModel):
  decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO","ABRIR_CHAMADO"]
  urgencia: Literal["BAIXA","MEDIA","ALTA"]
  campos_faltantes: List[str]= Field(default_factory=list)


llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key=GOOGLE_API_KEY
)

from langchain_core.messages import SystemMessage, HumanMessage

triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem: str) -> Dict:
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])
  
    return saida.model_dump()

!pip install -q --upgrade langchain_community faiss-cpu langchain-text-splitters pymupdf

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

docs = []
# Substitua este caminho pelo caminho real do seu arquivo PDF
file_path = Path("/content/Politicas_Connect_Gamers.pdf")

if file_path.exists():
    try:
        loader = PyMuPDFLoader(str(file_path))
        docs.extend(loader.load())
        print(f"Carregado com sucesso arquivo {file_path.name}")
    except Exception as e:
        print(f"Erro ao carregar arquivo {file_path.name}: {e}")
else:
    print(f"Arquivo não encontrado: {file_path.name}. Por favor, verifique o caminho.")

print(f"Total de documentos carregados: {len(docs)}")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)

chunks = splitter.split_documents(docs)

for chunk in chunks:
 print(chunk)
 print("**************")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                     search_kwargs={"score_threshold":0.3, "k": 4})


from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda apenas 'Não sei'."),

    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

def perguntar_politica_RAG(pergunta: str) -> Dict:
    docs_relacionados = retriever.invoke(pergunta)

    if not docs_relacionados:
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    answer = document_chain.invoke({"input": pergunta,
                                    "context": docs_relacionados})

    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    return {"answer": txt,
            "citacoes": formatar_citacoes(docs_relacionados, pergunta),
            "contexto_encontrado": True}

testes = ["Quais os requisitos de senha?",
          "Quais as politicas de segurança?",
          "O que é a Connect Gamers?",
          "Quais as politicas de uso?"]

for msg_teste in testes:
    resposta = perguntar_politica_RAG(msg_teste)
    print(f"PERGUNTA: {msg_teste}")
    print(f"RESPOSTA: {resposta['answer']}")
    if resposta['contexto_encontrado']:
        print("CITAÇÕES:")
        for c in resposta['citacoes']:
            print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
            print(f"   Trecho: {c['trecho']}")
        print("------------------------------------")

!pip install -q --upgrade langgraph


from typing import TypedDict, Optional

class AgentState(TypedDict, total = False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str

def node_triagem(state: AgentState) -> AgentState:
    print("Executando nó de triagem...")
    return {"triagem": triagem(state["pergunta"])}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("Executando nó de auto_resolver...")
    resposta_rag = perguntar_politica_RAG(state["pergunta"])

    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }

    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"

    return update


def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó de pedir_info...")
    faltantes = state["triagem"].get("campos_faltantes", [])
    if faltantes:
        detalhe = ",".join(faltantes)
    else:
        detalhe = "Tema e contexto específico"

    return {
        "resposta": f"Para avançar, preciso que detalhe: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }

KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]

    if decisao == "AUTO_RESOLVER": return "auto"
    if decisao == "PEDIR_INFO": return "info"
    if decisao == "ABRIR_CHAMADO": return "chamado"


def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("Decidindo após o auto_resolver...")

    if state.get("rag_sucesso"):
        print("Rag com sucesso, finalizando o fluxo.")
        return "ok"

    state_da_pergunta = (state["pergunta"] or "").lower()

    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print("Rag falhou, mas foram encontradas keywords de abertura de ticket. Abrindo...")
        return "chamado"

    print("Rag falhou, sem keywords, vou pedir mais informações...")
    return "info"


from langgraph.graph import StateGraph, START, END

workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado"
})

workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "ok": END
})

workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

# 6. `compile()` transforma nosso mapa de desenho em um objeto que o computador consegue executar.
grafo = workflow.compile()
