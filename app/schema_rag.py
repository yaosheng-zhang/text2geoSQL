import logging
import psycopg
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from app.config import bge_embeddings, DB_URL

logger = logging.getLogger(__name__)
_vectorstore = None

# 需要忽略的底层系统表（通用于绝大多数 PostgreSQL/PostGIS 库）
IGNORE_TABLES = {'spatial_ref_sys', 'geometry_columns', 'geography_columns'}

def get_dynamic_m_schema_docs() -> list[Document]:
    docs = []
    try:
        logger.info("正在执行通用 M-Schema 自动化构建...")
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                
                # schema_dict 结构: table_name -> { "cols": [], "pk": None, "fks_out": [], "fks_in": [] }
                schema_dict = defaultdict(lambda: {"cols": [], "pk": None, "fks_out": [], "fks_in": []})
                has_spatial_data = False # 动态探测是否包含空间数据
                
                # ==========================================
                # 1. 提取全库表结构与字段类型
                # ==========================================
                cur.execute("""
                    SELECT table_name, column_name, data_type, udt_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                """)
                for t_name, c_name, d_type, udt_name in cur.fetchall():
                    if t_name in IGNORE_TABLES:
                        continue
                    schema_dict[t_name]["cols"].append({"name": c_name, "type": d_type})
                    if udt_name.lower() in ('geometry', 'geography'):
                        has_spatial_data = True

                # ==========================================
                # 2. 提取物理主键 (Primary Keys)
                # ==========================================
                cur.execute("""
                    SELECT kcu.table_name, kcu.column_name
                    FROM information_schema.table_constraints tco
                    JOIN information_schema.key_column_usage kcu 
                      ON kcu.constraint_name = tco.constraint_name 
                    WHERE tco.constraint_type = 'PRIMARY KEY' AND tco.table_schema = 'public'
                """)
                for t_name, c_name in cur.fetchall():
                    if t_name in schema_dict:
                        schema_dict[t_name]["pk"] = c_name

                # ==========================================
                # 3. 提取物理外键 (Explicit Foreign Keys)
                # ==========================================
                explicit_links = set() # 记录已有的明确关联，防止隐式推断重复
                
                cur.execute("""
                    SELECT tc.table_name, kcu.column_name, ccu.table_name, ccu.column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
                """)
                for c_table, c_col, p_table, p_col in cur.fetchall():
                    if c_table in schema_dict and p_table in schema_dict:
                        schema_dict[c_table]["fks_out"].append({"my_col": c_col, "target_table": p_table, "target_col": p_col, "type": "物理外键"})
                        schema_dict[p_table]["fks_in"].append({"source_table": c_table, "source_col": c_col, "my_col": p_col, "type": "物理外键"})
                        explicit_links.add(f"{c_table}.{c_col}->{p_table}.{p_col}")

                # ==========================================
                # 4. 启发式推断隐式外键 (Implicit Foreign Keys)
                # ==========================================
                # 核心逻辑：如果 A表 的字段名等于 B表的主键名，或者等于 "B表名单数_主键名"
                for t_name, info in schema_dict.items():
                    for col in info["cols"]:
                        c_name = col["name"]
                        
                        # 不把自己的主键当成外键
                        if c_name == info["pk"]:
                            continue
                            
                        # 遍历其他所有表寻找匹配的主键
                        for target_t_name, target_info in schema_dict.items():
                            if t_name == target_t_name or not target_info["pk"]:
                                continue
                                
                            target_pk = target_info["pk"]
                            target_t_singular = target_t_name.rstrip('s') # 简单的去复数处理
                            
                            # 匹配规则 1: 字段名完全等于目标表主键名 (例如 chunk_id == chunk_id)
                            # 匹配规则 2: 字段名 == 目标表名_主键 (例如 user_id == user.id)
                            is_match = (c_name == target_pk) or (c_name == f"{target_t_singular}_{target_pk}")
                            
                            if is_match:
                                link_key = f"{t_name}.{c_name}->{target_t_name}.{target_pk}"
                                # 如果物理外键里没定义，我们把它作为逻辑外键加进去
                                if link_key not in explicit_links:
                                    schema_dict[t_name]["fks_out"].append({"my_col": c_name, "target_table": target_t_name, "target_col": target_pk, "type": "逻辑推断外键"})
                                    schema_dict[target_t_name]["fks_in"].append({"source_table": t_name, "source_col": c_name, "my_col": target_pk, "type": "逻辑推断外键"})
                                    explicit_links.add(link_key)

                # ==========================================
                # 5. 渲染通用 M-Schema 知识块文本
                # ==========================================
                for table_name, info in schema_dict.items():
                    lines = [f"Table: {table_name}"]
                    lines.append("Columns:")
                    
                    for col in info["cols"]:
                        pk_mark = " [PRIMARY KEY]" if col["name"] == info["pk"] else ""
                        lines.append(f"  - {col['name']} ({col['type']}){pk_mark}")
                    
                    # 渲染入度与出度关联 (Text2SQL 的关键)
                    if info["fks_out"] or info["fks_in"]:
                        lines.append("Relationships (JOIN Paths):")
                        
                        # 告诉大模型怎么从这张表 JOIN 出去
                        for fk in info["fks_out"]:
                            lines.append(f"  - To connect to [{fk['target_table']}], use: JOIN {fk['target_table']} ON {table_name}.{fk['my_col']} = {fk['target_table']}.{fk['target_col']} ({fk['type']})")
                        
                        # 告诉大模型别人怎么 JOIN 进这张表
                        for fk in info["fks_in"]:
                            lines.append(f"  - [{fk['source_table']}] can connect here using: JOIN {fk['source_table']} ON {fk['source_table']}.{fk['source_col']} = {table_name}.{fk['my_col']} ({fk['type']})")

                    # 创建文档
                    docs.append(Document(page_content="\n".join(lines), metadata={"table": table_name}))

                logger.info("成功构建 %d 个表的通用 M-Schema 知识块", len(docs))

                # ==========================================
                # 6. 动态通用 SQL/空间函数字典 (不写死表名)
                # ==========================================
                # 只有当数据库里真的探测到 geometry 字段时，才附加 PostGIS 通用知识
                if has_spatial_data:
                    postgis_doc = """
General SQL & PostGIS Knowledge (No specific tables):
- Use `JOIN` strictly according to the 'Relationships' paths provided in table schemas.
- Spatial Intersects / Contains: `ST_Contains(polygon_geom, point_geom)` or `ST_Intersects(geom1, geom2)`
- Spatial Distance: `ST_DWithin(geom1, geom2, distance)`
- Always ensure you are JOINing a table that contains a GEOMETRY column before applying spatial functions.
"""
                    docs.append(Document(page_content=postgis_doc.strip(), metadata={"table": "dialect_guide"}))

    except Exception as e:
        logger.error("自动构建 M-Schema 失败: %s", e, exc_info=True)

    return docs

def get_schema_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        logger.info("正在初始化 Schema RAG 内存向量库...")
        try:
            schema_docs = get_dynamic_m_schema_docs()
            _vectorstore = InMemoryVectorStore.from_documents(
                documents=schema_docs,
                embedding=bge_embeddings,
            )
            logger.info("Schema RAG 初始化完成，共入库 %d 个语义知识块", len(schema_docs))
        except Exception as e:
            logger.error("Schema RAG 初始化失败", exc_info=True)
            raise
    return _vectorstore