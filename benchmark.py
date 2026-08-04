import csv
import json
import os
import re
import time
from datetime import datetime
from itertools import product

from rag import RAGPipeline


# Add more datasets here as you grow experiments.
DATASETS = [
    {
        "name": "hall_allotment",
        "pdf_path": "data/1e603f08c8d972ef_2nd yr BOYS_HOSTEL_UG2025.pdf",
        "question_file": "data/questions/hall_allotment.json",
    },
        {
        "name": "git",
        "pdf_path": "data/6103fbea51a6bef6_progit-8-105.pdf",
        "question_file": "data/questions/git.json",
    },
    {
        "name": "academic_rules",
        "pdf_path": "data/ed8e359dbb85db7a_regulation_UG_corrected.pdf",
        "question_file": "data/questions/academic_rules.json",
    },
]

RESULT_DIR = "benchmark_results"

# Experiment grid
RETRIEVERS = ["mmr", "similarity"]
K_VALUES = [5, 10, 20]
CHUNK_SIZES = [256, 512, 1024]
EMBEDDING_MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"]


def create_experiments():
    experiments = []
    for retriever, k, chunk_size, embedding in product(
        RETRIEVERS,
        K_VALUES,
        CHUNK_SIZES,
        EMBEDDING_MODELS,
    ):
        chunk_overlap = max(1, chunk_size // 10)
        fetch_k = max(10, k * 2)
        name = (
            f"{retriever}|k={k}|fetch_k={fetch_k}|chunk={chunk_size}|"
            f"overlap={chunk_overlap}|emb={embedding}"
        )
        experiments.append(
            {
                "name": name,
                "retriever_type": retriever,
                "k": k,
                "fetch_k": fetch_k,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "embedding_model": embedding,
            }
        )
    return experiments


def check_retrieval(context, expected):
    retrieved_text = " ".join(doc.page_content for doc in context)
    normalized_retrieved = normalize_text(retrieved_text)
    expected_variants = parse_expected_variants(expected)

    for variant in expected_variants:
        normalized_variant = normalize_text(variant)
        if not normalized_variant:
            continue
        if normalized_variant in normalized_retrieved:
            return True

        variant_tokens = token_set(normalized_variant)
        if not variant_tokens:
            continue

        retrieved_tokens = token_set(normalized_retrieved)
        overlap = len(variant_tokens & retrieved_tokens)
        coverage = overlap / len(variant_tokens)
        if coverage >= 0.6:
            return True

    return False


def check_answer(answer, expected):
    normalized_answer = normalize_text(answer)
    for variant in parse_expected_variants(expected):
        normalized_variant = normalize_text(variant)
        if normalized_variant and normalized_variant in normalized_answer:
            return True
    return False


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def token_set(text):
    return {token for token in re.split(r"[^a-z0-9]+", text) if token}


def parse_expected_variants(expected):
    # Support alternative valid answers using "||", e.g. "rp hall||ram prasad hall".
    return [part.strip() for part in expected.split("||") if part.strip()]


def is_quota_error(error):
    error_text = str(error).lower()
    quota_markers = [
        "resource_exhausted",
        "resource exhausted",
        "429",
        "quota",
        "rate limit",
        "rate-limit",
        "retry delay",
    ]
    return any(marker in error_text for marker in quota_markers)


def invoke_with_retry(chain, question, max_retries=3, delay_seconds=70):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke({"input": question}), None
        except Exception as error:
            last_error = error
            if not is_quota_error(error) or attempt == max_retries:
                return None, error

            print(
                f"Quota/rate limit hit. Retrying in {delay_seconds} seconds "
                f"(attempt {attempt}/{max_retries})..."
            )
            time.sleep(delay_seconds)

    return None, last_error


def load_questions(question_file):
    with open(question_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Dataset: {data.get('dataset', 'unknown')}")
    questions = data.get("questions", [])
    print(f"Questions: {len(questions)}")
    return questions


def load_question_data(question_file):
    with open(question_file, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_dataset(dataset):
    if not os.path.exists(dataset["pdf_path"]):
        raise FileNotFoundError(f"Missing PDF: {dataset['pdf_path']}")
    if not os.path.exists(dataset["question_file"]):
        raise FileNotFoundError(f"Missing question file: {dataset['question_file']}")

    question_data = load_question_data(dataset["question_file"])
    question_dataset_name = question_data.get("dataset")
    if question_dataset_name != dataset["name"]:
        raise ValueError(
            "Dataset name mismatch: "
            f"DATASETS entry is '{dataset['name']}' but "
            f"question file declares '{question_dataset_name}' "
            f"({dataset['question_file']})"
        )

    questions = question_data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No questions found in {dataset['question_file']}")

    for index, item in enumerate(questions):
        if "question" not in item or "expected_answer" not in item:
            raise ValueError(
                f"Invalid question at index {index} in "
                f"{dataset['question_file']}: requires 'question' "
                "and 'expected_answer'"
            )


def run_benchmark():
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    all_results = []
    experiments = create_experiments()
    chain_cache = {}

    for dataset in DATASETS:
        validate_dataset(dataset)
        print("\n############################################")
        print(f"Running dataset: {dataset['name']}")
        print("############################################")
        questions = load_questions(dataset["question_file"])

        for experiment in experiments:
            print("\n====================")
            print("Running:", experiment["name"])
            print("====================")

            cache_key = (
                dataset["pdf_path"],
                experiment["name"],
            )
            indexing_time = 0.0
            index_source = "cache"

            if cache_key in chain_cache:
                chain = chain_cache[cache_key]
            else:
                rag = RAGPipeline(
                    chunk_size=experiment["chunk_size"],
                    chunk_overlap=experiment["chunk_overlap"],
                    embedding_model=experiment["embedding_model"],
                    retriever_type=experiment["retriever_type"],
                    k=experiment["k"],
                    fetch_k=experiment["fetch_k"],
                )

                start_index = time.time()
                chain = rag.process_pdf(dataset["pdf_path"])
                indexing_time = round(time.time() - start_index, 3)
                index_source = "built"
                chain_cache[cache_key] = chain

            print("Indexing time:", indexing_time, f"({index_source})")

            for item in questions:
                question = item["question"]
                expected = item["expected_answer"]

                print("\nQuestion:", question)
                start = time.time()

                error_message = ""
                answer = ""
                context = []
                retrieved_chunk_count = 0
                retrieval_score = False
                answer_score = False

                try:
                    response, retry_error = invoke_with_retry(chain, question)
                    if retry_error is not None:
                        raise retry_error

                    answer = response.get("answer", "")
                    context = response.get("context", [])
                    retrieved_chunk_count = len(context)
                    retrieval_score = check_retrieval(context, expected)
                    answer_score = check_answer(answer, expected)
                except Exception as error:
                    error_message = str(error)

                response_time = round(time.time() - start, 3)

                all_results.append(
                    {
                        "run_timestamp": run_timestamp,
                        "dataset": dataset["name"],
                        "pdf_path": dataset["pdf_path"],
                        "question_file": dataset["question_file"],
                        "experiment": experiment["name"],
                        "retriever": experiment["retriever_type"],
                        "k": experiment["k"],
                        "fetch_k": experiment["fetch_k"],
                        "chunk_size": experiment["chunk_size"],
                        "chunk_overlap": experiment["chunk_overlap"],
                        "embedding": experiment["embedding_model"],
                        "indexing_time_sec": indexing_time,
                        "index_source": index_source,
                        "question": question,
                        "expected": expected,
                        "answer": answer,
                        "retrieved_chunk_count": retrieved_chunk_count,
                        "retrieval_correct": retrieval_score,
                        "answer_correct": answer_score,
                        "response_time_sec": response_time,
                        "error": error_message,
                    }
                )

                if error_message:
                    print("Error:", error_message)
                else:
                    print("Answer:", answer)
                    print("Retrieval:", retrieval_score)
                    print("Correct:", answer_score)

    save_results(all_results, run_timestamp)


def save_results(results, run_timestamp):
    if not results:
        print("No benchmark results to save.")
        return

    os.makedirs(RESULT_DIR, exist_ok=True)

    safe_ts = run_timestamp.replace(":", "-")
    run_file = os.path.join(RESULT_DIR, f"results_{safe_ts}.csv")
    latest_file = os.path.join(RESULT_DIR, "results_latest.csv")

    fieldnames = list(results[0].keys())
    for file_path in [run_file, latest_file]:
        with open(file_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nSaved results to {run_file}")
    print(f"Updated latest snapshot at {latest_file}")


if __name__ == "__main__":
    run_benchmark()
