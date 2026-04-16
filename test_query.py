# test_query.py - 带健康检查和诊断的测试脚本
import asyncio
import httpx
import traceback
import sys

BASE_URL = "http://127.0.0.1:8000"


async def wait_for_service(client: httpx.AsyncClient, max_wait: int = 30):
    """等待服务启动，最多等待 max_wait 秒"""
    print(f"\n⏳ 正在等待服务启动 ({BASE_URL}) ...")
    print(f"   (最多等待 {max_wait} 秒)\n")

    for i in range(max_wait):
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                db_status = data.get("database", "unknown")
                status = data.get("status", "unknown")
                print(f"\n✅ 服务已就绪!")
                print(f"   状态: {status}")
                print(f"   数据库: {db_status}")
                if db_status != "connected":
                    print(f"\n⚠️  警告: 数据库未连接!")
                    print(f"   服务启动成功，但数据库连接失败")
                    print(f"   请检查:")
                    print(f"   1. PostgreSQL 服务是否运行")
                    print(f"   2. 数据库连接信息是否正确")
                    print(f"   3. 查看服务日志了解详细错误\n")
                return True
        except httpx.ConnectError:
            pass
        except Exception as e:
            if i == 0:
                print(f"   健康检查异常: {type(e).__name__}")

        if (i + 1) % 5 == 0:
            print(f"   已等待 {i + 1}s，服务尚未就绪...")
        await asyncio.sleep(1)

    print(f"\n❌ 超时: 服务在 {max_wait}s 内未启动")
    print("\n📋 诊断步骤:")
    print(f"   1. 确认服务已启动:")
    print(f"      python -m app.main")
    print(f"   2. 检查端口 8000 是否被占用:")
    print(f"      netstat -ano | findstr :8000  (Windows)")
    print(f"      lsof -i :8000  (Mac/Linux)")
    print(f"   3. 查看服务启动日志排查错误")
    print(f"   4. 确认 PostgreSQL 服务已启动\n")
    return False


async def test():
    print("=" * 70)
    print("🌍 空间知识库 Text2GeoSQL 测试")
    print(f"📍 目标地址: {BASE_URL}")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 先等待服务就绪
        if not await wait_for_service(client):
            sys.exit(1)

        queries = [
            # "涩谷行政区内评级高于 4.5 的旅游景区有哪些？",
            "各用地类型最大允许建筑密度统计",
            # "日本东京塔在哪个行政区域，精确的经纬度是多少？"
        ]

        success_count = 0
        for i, q in enumerate(queries, 1):
            print(f"\n--- 测试 {i}/{len(queries)} ---")
            print(f"查询: {q}")

            try:
                resp = await client.post(
                    f"{BASE_URL}/query",
                    json={"query": q},
                    timeout=120.0,  # LLM 调用可能较慢
                )  

                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    print(f"✅ 成功! 返回 {len(results)} 条结果")
                    print(f"SQL: {data.get('sql', 'N/A')[:500]}")
                    for item in results[:3]:
                        print(f"  -> {item.get('title', 'N/A')}")
                    success_count += 1
                else:
                    print(f"❌ HTTP {resp.status_code}: {resp.text[:300]}")

            except httpx.ReadTimeout:
                print("⏱️  请求超时 (>120s)，LLM 调用可能过慢")
            except Exception as e:
                print(f"❌ 异常: {e}")
                traceback.print_exc()

        print(f"\n{'=' * 70}")
        print(f"📊 测试完成: {success_count}/{len(queries)} 成功")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test())