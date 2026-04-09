-- =============================================
-- 工业级空间知识库初始化脚本 (init.sql)
-- 包含：主表 + 测试数据 + 多表扩展 + 动态向量实体表
-- =============================================

-- 启用必要扩展
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- 用于模糊匹配后备

-- =============================================
-- 1. 主空间分块表
-- =============================================
CREATE TABLE IF NOT EXISTS spatial_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    embedding VECTOR(1024),
    title TEXT,
    source TEXT,
    geometry GEOMETRY(Geometry, 4326) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 主表索引（生产必备）
CREATE INDEX IF NOT EXISTS idx_embedding ON spatial_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_geometry ON spatial_chunks USING gist (geometry);
CREATE INDEX IF NOT EXISTS idx_metadata ON spatial_chunks USING gin (metadata);

-- =============================================
-- 2. 测试数据：东京真实地标（Point + Polygon）
-- =============================================
INSERT INTO spatial_chunks (title, content, geometry, metadata, source) 
VALUES
('涩谷站', '涩谷站是东京最繁忙的交通枢纽之一，周边有忠犬八公像、涩谷 Scramble 交叉口，是东京潮流文化中心。', ST_GeomFromText('POINT(139.7017 35.6580)', 4326), '{"city": "Tokyo", "ward": "Shibuya", "type": "station", "radius_m": 800}', 'OSM'),
('明治神宫', '明治神宫是东京最大的神社，占地70万平方米，森林茂密，是祈福和婚礼圣地。', ST_GeomFromText('POINT(139.6993 35.6764)', 4326), '{"city": "Tokyo", "ward": "Shibuya", "type": "shrine", "radius_m": 1200}', 'OSM'),
('东京铁塔', '东京铁塔高333米，是东京标志性建筑，夜景极美，可俯瞰整个东京。', ST_GeomFromText('POINT(139.7454 35.6586)', 4326), '{"city": "Tokyo", "ward": "Minato", "type": "landmark", "radius_m": 600}', 'OSM'),
('浅草寺', '浅草寺是东京最古老的寺庙，雷门和仲见世通り闻名世界，每年吸引数千万游客。', ST_GeomFromText('POINT(139.7967 35.7147)', 4326), '{"city": "Tokyo", "ward": "Taito", "type": "temple", "radius_m": 900}', 'OSM'),
('新宿站', '新宿站是世界客流量最大的车站，周边高楼林立，是东京商业与娱乐中心。', ST_GeomFromText('POINT(139.7005 35.6895)', 4326), '{"city": "Tokyo", "ward": "Shinjuku", "type": "station", "radius_m": 1000}', 'OSM'),
('秋叶原', '秋叶原是全球著名的电子产品和动漫圣地，女仆咖啡厅和游戏中心密集。', ST_GeomFromText('POINT(139.7730 35.6987)', 4326), '{"city": "Tokyo", "ward": "Chiyoda", "type": "district", "radius_m": 700}', 'OSM'),
('上野公园', '上野公园是东京最大的公园，樱花季著名，内有博物馆和动物园。', ST_GeomFromText('POINT(139.7714 35.7138)', 4326), '{"city": "Tokyo", "ward": "Taito", "type": "park", "radius_m": 1500}', 'OSM'),
('皇居', '皇居是日本天皇住所，周边护城河和东御苑是市民休闲场所。', ST_GeomFromText('POINT(139.7530 35.6852)', 4326), '{"city": "Tokyo", "ward": "Chiyoda", "type": "landmark", "radius_m": 2000}', 'OSM'),
('银座', '银座是东京最高档的购物和餐饮区，集中了众多奢侈品牌和米其林餐厅。', ST_GeomFromText('POINT(139.7650 35.6719)', 4326), '{"city": "Tokyo", "ward": "Chuo", "type": "district", "radius_m": 800}', 'OSM'),
('东京天空树', '东京天空树高634米，是世界最高的自立式电波塔，可360度观景。', ST_GeomFromText('POINT(139.8107 35.7101)', 4326), '{"city": "Tokyo", "ward": "Sumida", "type": "landmark", "radius_m": 1000}', 'OSM');

-- 示例 Polygon（历史街区）
INSERT INTO spatial_chunks (title, content, geometry, metadata, source) 
VALUES
('涩谷历史街区', '涩谷道玄坂和宇田川町一带保留了部分战后建筑，是探索东京老街的好去处。', 
 ST_GeomFromText('POLYGON((139.700 35.655, 139.705 35.655, 139.705 35.660, 139.700 35.660, 139.700 35.655))', 4326),
 '{"city": "Tokyo", "ward": "Shibuya", "type": "district"}', 'OSM');

-- =============================================
-- 3. 扩展表1：POI 详细信息（属性表）
-- =============================================
CREATE TABLE IF NOT EXISTS poi_details (
    chunk_id UUID PRIMARY KEY REFERENCES spatial_chunks(id) ON DELETE CASCADE,
    description TEXT,
    visit_count INT,
    rating NUMERIC(3,1),
    category TEXT,
    tags TEXT[],
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- 插入 POI 测试数据
INSERT INTO poi_details (chunk_id, description, visit_count, rating, category, tags) 
SELECT id, '热门旅游景点，适合拍照和购物', 5000000, 4.8, 'tourist', ARRAY['shopping','photo'] 
FROM spatial_chunks 
WHERE title IN ('涩谷站', '明治神宫', '浅草寺')
ON CONFLICT (chunk_id) DO NOTHING;

-- =============================================
-- 4. 扩展表2：行政区划（Polygon）
-- =============================================
CREATE TABLE IF NOT EXISTS admin_boundaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ward_name TEXT NOT NULL,
    district TEXT,
    geometry GEOMETRY(MultiPolygon, 4326),
    population INT,
    area_sqm NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_admin_geom ON admin_boundaries USING gist (geometry);

-- 插入行政区测试数据
INSERT INTO admin_boundaries (ward_name, district, geometry, population, area_sqm) 
VALUES
('Shibuya', 'Tokyo', ST_GeomFromText('MULTIPOLYGON(((139.69 35.65,139.71 35.65,139.71 35.67,139.69 35.67,139.69 35.65)))', 4326), 230000, 15000000),
('Minato', 'Tokyo', ST_GeomFromText('MULTIPOLYGON(((139.73 35.65,139.76 35.65,139.76 35.67,139.73 35.67,139.73 35.65)))', 4326), 260000, 20000000)
ON CONFLICT DO NOTHING;

-- =============================================
-- 5. 扩展表3：基础设施
-- =============================================
CREATE TABLE IF NOT EXISTS infrastructure (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT,
    type TEXT,           -- 'subway', 'bus', 'road'
    geometry GEOMETRY(Geometry, 4326),
    capacity INT,
    status TEXT
);

CREATE INDEX IF NOT EXISTS idx_infra_geom ON infrastructure USING gist (geometry);

-- 插入基础设施测试数据
INSERT INTO infrastructure (name, type, geometry, capacity, status) 
VALUES
('涩谷地铁站', 'subway', ST_GeomFromText('POINT(139.7017 35.6580)', 4326), 1000000, 'active'),
('东京塔附近公交站', 'bus', ST_GeomFromText('POINT(139.745 35.658)', 4326), 5000, 'active')
ON CONFLICT DO NOTHING;

-- =============================================
-- 6. 动态实体向量表（核心，用于通用实体 Grounding）
-- =============================================
CREATE TABLE IF NOT EXISTS value_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_table TEXT NOT NULL,
    source_column TEXT NOT NULL,
    raw_value TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    entity_type TEXT NOT NULL,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_value_embedding ON value_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_value_lookup ON value_embeddings (source_table, source_column, entity_type);
CREATE INDEX IF NOT EXISTS idx_raw_value_trgm ON value_embeddings USING gin (raw_value gin_trgm_ops);

-- =============================================
-- 初始化完成提示
-- =============================================
DO $$
BEGIN
    RAISE NOTICE '✅ 工业级空间知识库数据库初始化完成！';
    RAISE NOTICE '请运行 etl_value_embeddings.py 生成动态实体向量。';
END $$;