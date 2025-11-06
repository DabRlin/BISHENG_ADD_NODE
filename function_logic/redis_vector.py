"""
Bisheng 平台 - Redis 向量检索节点（生产版本）

输入：
- arg1: 用户查询问题 (user_input)
- arg2: 文件内容 (dialog_files_content)

输出：
- dict: {'chunks': list[str]}  # 最相关的文本块列表
"""

import requests
import redis
import numpy as np
from typing import List, Dict, Any
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.query import Query


def main(arg1: str, arg2: str) -> dict:
    """Bisheng 平台主函数"""

    # ==================== 配置区 ====================
    API_KEY = "sk-mkvgdxunnjdrrjxrlfulkizzxkiyapgxqeaqokwlxxrrxlkr"
    BASE_URL = "https://api.siliconflow.cn/v1"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

    REDIS_HOST = "192.168.0.243"
    REDIS_PORT = 6380
    REDIS_DB = 0
    REDIS_PASSWORD = None
    REDIS_INDEX_NAME = "bisheng_docs_idx"

    VECTOR_DIM = 1024
    CHUNK_SIZE = 250
    OVERLAP_SIZE = 50
    TOP_K = 3
    USE_RERANK = True

    # ==================== 内部函数定义 ====================

    def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
        """将文本分割成多个块"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    def get_embeddings(chunks: List[str]) -> List[List[float]]:
        """对文本块进行向量化"""
        url = f"{BASE_URL}/embeddings"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": EMBEDDING_MODEL,
            "input": chunks,
            "encoding_format": "float"
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            result = response.json()

            if response.status_code == 200:
                return [item['embedding'] for item in result['data']]
            return []

        except Exception:
            return []

    def connect_redis() -> redis.Redis:
        """连接到 Redis"""
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=False
        )
        r.ping()
        return r

    def create_vector_index(redis_client: redis.Redis, index_name: str) -> bool:
        """创建 Redis 向量索引"""
        try:
            # 删除旧索引
            try:
                redis_client.ft(index_name).dropindex(delete_documents=True)
            except Exception:
                pass

            schema = (
                TextField("content"),
                NumericField("chunk_id"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            )

            redis_client.ft(index_name).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
            )
            return True

        except Exception:
            return False

    def store_vectors_to_redis(redis_client: redis.Redis, chunks: List[str],
                               embeddings: List[List[float]]) -> int:
        """将向量存储到 Redis"""
        stored_count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                redis_client.hset(
                    f"doc:{i}",
                    mapping={
                        "content": chunk,
                        "chunk_id": i,
                        "embedding": embedding_bytes
                    }
                )
                stored_count += 1
            except Exception:
                pass
        return stored_count

    def vector_search_redis(redis_client: redis.Redis, query_vector: List[float],
                            top_k: int) -> List[Dict[str, Any]]:
        """在 Redis 中执行向量搜索"""
        try:
            query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
            query = (
                Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
                .sort_by("score")
                .return_fields("chunk_id", "content", "score")
                .dialect(2)
            )

            results = redis_client.ft(REDIS_INDEX_NAME).search(
                query,
                query_params={"vec": query_bytes}
            )

            search_results = []
            for doc in results.docs:
                search_results.append({
                    "chunk_id": int(doc.chunk_id),
                    "content": doc.content,
                    "score": float(doc.score)
                })

            return search_results

        except Exception:
            return []

    def rerank_documents(query: str, documents: List[str], top_n: int) -> Dict[str, Any]:
        """使用 Rerank 模型重新排序"""
        url = f"{BASE_URL}/rerank"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents))
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            result = response.json()
            if response.status_code == 200:
                return result
            return {"results": []}

        except Exception:
            return {"results": []}

    # ==================== 主逻辑 ====================

    # 参数验证
    if not arg1 or not arg1.strip():
        return {'chunks': []}
    if not arg2 or not arg2.strip():
        return {'chunks': []}

    query = arg1.strip()
    file_content = arg2.strip()

    try:
        # 1. 文本分块
        chunks = split_text_into_chunks(file_content, CHUNK_SIZE, OVERLAP_SIZE)
        if not chunks:
            return {'chunks': []}

        # 2. 生成 Embeddings
        embeddings = get_embeddings(chunks)
        if not embeddings:
            return {'chunks': []}

        # 3. 连接 Redis
        redis_client = connect_redis()

        # 4. 创建索引并存储向量
        create_vector_index(redis_client, REDIS_INDEX_NAME)
        stored_count = store_vectors_to_redis(redis_client, chunks, embeddings)
        if stored_count == 0:
            return {'chunks': []}

        # 5. 查询向量化
        query_embeddings = get_embeddings([query])
        if not query_embeddings:
            return {'chunks': []}
        query_vector = query_embeddings[0]

        # 6. 执行检索
        if USE_RERANK:
            # Redis + Rerank 组合
            initial_top_k = min(TOP_K * 3, len(chunks))
            redis_results = vector_search_redis(redis_client, query_vector, initial_top_k)

            if not redis_results:
                return {'chunks': []}

            candidate_chunks = [chunks[r['chunk_id']] for r in redis_results]
            rerank_result = rerank_documents(query, candidate_chunks, TOP_K)

            result_chunks = []
            for item in rerank_result.get('results', [])[:TOP_K]:
                original_chunk_id = redis_results[item['index']]['chunk_id']
                result_chunks.append(chunks[original_chunk_id])

        else:
            # 仅使用 Redis 向量搜索
            redis_results = vector_search_redis(redis_client, query_vector, TOP_K)
            result_chunks = [r['content'] for r in redis_results]

        # 7. 返回结果
        return {'chunks': result_chunks}

    except Exception:
        return {'chunks': []}
