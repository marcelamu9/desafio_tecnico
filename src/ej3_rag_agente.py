import pandas as pd
from rank_bm25 import BM25Okapi
import json
import re


## carga y normalizacion 
def load_and_normalize_dataset(path: str, n_rows: int | None = 300) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.sample(n = 200, random_state = 42)

    cols_keep = [
        "laptop_id",
        "full_name",
        "link_profile",
        "producer",
        "price_in_dollar",
        "model",
        "ram",
        "disc",
        "display_resolution",
        "cpu",
        "gpu",
    ]
    df = df[cols_keep]

    if n_rows is not None:
        df = df.head(n_rows)


    return df

##  construccion de chunk
def build_chunks(df: pd.DataFrame):
    chunks_texts = []
    chunks_meta = []

    fields = [
        "producer",
        "full_name",
        "price_in_dollar",
        "model",
        "ram",
        "disc",
        "display_resolution",
        "cpu",
        "gpu",
    ]

    for _, row in df.iterrows():
        laptop_id = row["laptop_id"]

        for field in fields:
            value = str(row[field])

            if value and value != "nan":

                # CHUNK MEJORADO
                text = (
                    f"{row['full_name']} "
                    f"{row['producer']} {row['model']} "
                    f"{field}: {value}"
                )

                chunks_texts.append(text)
                chunks_meta.append(
                    {
                        "laptop_id": int(laptop_id),
                        "field": field,
                        "value": value,
                        "text": text,
                    }
                )
    return chunks_texts, chunks_meta





def build_bm25_index(chunks_texts):
    tokenized_corpus = [t.lower().split() for t in chunks_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

## retrieve chunks

def retrieve_chunks(query: str, bm25: BM25Okapi, tokenized_corpus, chunks_meta, k: int = 5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    # índices de los mejores k
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    retrieved = [chunks_meta[i] for i in top_idx]
    return retrieved

## Generate - respuesta preliominar

def guess_field_from_query(query: str) -> str | None:
    q = query.lower()
    if "ram" in q or "memoria" in q:
        return "ram"   # exactamente igual a la columna
    if "procesador" in q or "cpu" in q:
        return "cpu"
    if "almacenamiento" in q or "disco" in q or "ssd" in q:
        return "disc"
    if "gpu" in q or "gráfica" in q or "grafica" in q:
        return "gpu"
    if "sistema operativo" in q or "os" in q:
        return "operating_system"  # si tuvieras esa columna
    return None

def choose_best_laptop_id(query: str, retrieved_chunks: list[dict]) -> int:
    q = query.lower()
    scored = []
    for ch in retrieved_chunks:
        value = str(ch["value"]).lower()
        tokens = value.split()
        score = sum(1 for t in tokens if t in q)
        scored.append((score, ch["laptop_id"]))
    if not scored:
        return retrieved_chunks[0]["laptop_id"]
    scored.sort(reverse=True)
    best_score, best_id = scored[0]
    if best_score == 0:
        return retrieved_chunks[0]["laptop_id"]
    return best_id



def format_citation(ch: dict) -> str:
    return f"[{ch['laptop_id']}:{ch['field']}]"


def generate_answer_simple(
    query: str,
    retrieved_chunks: list[dict],
    chunks_meta: list[dict],
) -> str:
    target_field = guess_field_from_query(query)


    # 1) Elegir laptop más probable
    best_laptop_id = choose_best_laptop_id(query, retrieved_chunks)

    # 2) Todos los chunks de esa laptop
    candidates = [ch for ch in chunks_meta if ch["laptop_id"] == best_laptop_id]

    # 3) Nombre bonito: full_name → model → producer+model
    name_chunk = next(
        (ch for ch in candidates if ch["field"] in ("full_name", "model")),
        None,
    )
    if name_chunk is not None:
        laptop_name = str(name_chunk["value"])
    else:
        producer = next(
            (ch["value"] for ch in candidates if ch["field"] == "producer"),
            "",
        )
        model = next(
            (ch["value"] for ch in candidates if ch["field"] == "model"),
            "",
        )
        laptop_name = f"{producer} {model}".strip()

    # 4) Buscar campo que pide el usuario (ram, cpu, etc.) de forma robusta
    field_chunk = None
    if target_field:
        # match exacto
        field_chunk = next(
            (
                ch
                for ch in candidates
                if ch["field"].lower() == target_field.lower()
            ),
            None,
        )
        # si no hay exacto, buscar por “contiene”
        if field_chunk is None:
            field_chunk = next(
                (
                    ch
                    for ch in candidates
                    if target_field.lower() in ch["field"].lower()
                ),
                None,
            )

    # 5) Si encontramos el campo → respuesta bonita (citas en la MISMA frase)
    if field_chunk is not None:
        value = field_chunk["value"]
        cit_field = format_citation(field_chunk)
        cit_name = format_citation(name_chunk) if name_chunk is not None else ""
        return (
            f"La {laptop_name} tiene {value} de {field_chunk['field']} "
            f"{cit_field}{cit_name}."
        )

    # 6) Fallback si no encontramos el campo: concatenamos algunos chunks
    parts = []
    for ch in retrieved_chunks[:3]:
        parts.append(f"{ch['field']}: {ch['value']} {format_citation(ch)}")
    return " / ".join(parts)


# Agente critico

def simple_sent_tokenize(text: str):
    # Divide por . ? ! seguidos de espacio, de forma simple
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Quita vacíos y espacios extra
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def critique_answer(answer: str, retrieved_chunks: list[dict]):
    sentences = simple_sent_tokenize(answer)
    supported = []
    unsupported = []

    # Conjunto de citas válidas basadas en los chunks recuperados
    valid_citations = {format_citation(ch) for ch in retrieved_chunks}

    for sent in sentences:
        # Buscar cosas como [741:ram]
        cites = re.findall(r"\[\d+:[A-Za-z_]+\]", sent)
        if any(c in valid_citations for c in cites):
            supported.append(sent)
        else:
            unsupported.append(sent)

    decision = "accept" if len(unsupported) == 0 else "regenerate"
    return {
        "decision": decision,
        "supported": supported,
        "unsupported": unsupported,
    }


def format_citation(ch: dict) -> str:
    return f"[{ch['laptop_id']}:{ch['field']}]"

## logs
def answer_question(query: str, bm25, tokenized_corpus, chunks_meta, log_file="agent_logs.jsonl"):
    retrieved = retrieve_chunks(query, bm25, tokenized_corpus, chunks_meta)

    attempt = 0
    while True:
        attempt += 1
        answer = generate_answer_simple(query, retrieved, chunks_meta)
        critique = critique_answer(answer, retrieved)

        if critique["decision"] == "regenerate":
            log_entry = {
                "query": query,
                "attempt": attempt,
                "answer": answer,
                "decision": critique["decision"],
            "unsupported": critique["unsupported"],
            }
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if critique["decision"] == "accept" or attempt >= 3:
            return answer