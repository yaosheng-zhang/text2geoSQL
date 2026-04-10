# test_query.py - 带健康检查和诊断的测试脚本
import asyncio
import httpx
import traceback
import sys

BASE_URL = "http://127.0.0.1:8000"


async def wait_for_service(client: httpx.AsyncClient, max_wait: int = 30):
    """等待服务启动，最多等待 max_wait 秒"""
    print(f"正在等待服务启动 ({BASE_URL}) ...")
    for i in range(max_wait):
        try:
            resp = await client.get(f"{BASE_URL}/health", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                print(f"服务已就绪! 状态: {data}")
                return True
        except httpx.ConnectError:
            pass
        except Exception as e:
            print(f"  健康检查异常: {e}")

        if (i + 1) % 5 == 0:
            print(f"  已等待 {i + 1}s，服务尚未就绪...")
        await asyncio.sleep(1)

    print(f"超时: 服务在 {max_wait}s 内未启动")
    print("请确认:")
    print(f"  1. 服务已启动: python -m app.main")
    print(f"  2. 端口 8000 未被占用")
    print(f"  3. 查看服务端日志排查启动错误")
    return False


async def test():
    print("=" * 60)
    print("空间知识库 Text2GeoSQL 测试")
    print(f"目标地址: {BASE_URL}")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 先等待服务就绪
        if not await wait_for_service(client):
            sys.exit(1)

        queries = [
            # "涩谷行政区内评级高于 4.5 的旅游景区有哪些？",
            "查询已竣工且高度超过100米的建筑",
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
                    print(f"成功! 返回 {len(results)} 条结果")
                    print(f"SQL: {data.get('sql', 'N/A')[:500]}")
                    for item in results[:3]:
                        print(f"  -> {item.get('title', 'N/A')}")
                    success_count += 1
                else:
                    print(f"HTTP {resp.status_code}: {resp.text[:300]}")

            except httpx.ReadTimeout:
                print("请求超时 (>120s)，LLM 调用可能过慢")
            except Exception as e:
                print(f"异常: {e}")
                traceback.print_exc()

        print(f"\n{'=' * 60}")
        print(f"测试完成: {success_count}/{len(queries)} 成功")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test())