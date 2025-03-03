import asyncio
import aiohttp
import json

# 定义 API 地址和端口
API_URL = "http://127.0.0.1:8222/v1/chat/completions"

# 定义异步请求函数
async def send_request(session, question, semaphore, progress, total):
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": "DeepSeek-R1-Distill-Qwen-7B",  # 与 --served-model-name 保持一致
        "messages": [
            {"role": "user", "content": question}
        ],
        "temperature": 0.6,  # 控制生成的多样性
        "top_p": 0.95,  # 样本采样的概率阈值
        "max_tokens": 7000  # 生成的最大 token 数
    }
    async with semaphore:  # 使用信号量限制并发
        try:
            async with session.post(API_URL, json=payload, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                else:
                    content = f"Error: {response.status} - {await response.text()}"
        except Exception as e:
            content = f"Request failed: {e}"
        
        # 实时更新进度
        progress["count"] += 1
        print(f"Processed {progress['count']} / {total} questions", end="\r")
        return content

# 主函数：处理数据并保存结果
async def process_data(input_file, output_file, concurrent_limit=10):
    results = []

    # 读取输入文件（逐行解析 JSON）
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)  # 总数据量
    progress = {"count": 0}  # 用于跟踪处理进度
    semaphore = asyncio.Semaphore(concurrent_limit)  # 定义并发限制

    async with aiohttp.ClientSession() as session:
        tasks = []
        for line in lines:
            record = json.loads(line)  # 解析每行 JSON
            question = record["input"]  # 提取问题
            # 创建异步任务
            task = send_request(session, question, semaphore, progress, total)
            tasks.append((record, task))

        # 并发发送请求
        responses = await asyncio.gather(*(task[1] for task in tasks))

        # 处理结果
        for (record, response) in zip((task[0] for task in tasks), responses):
            record["output"] = response  # 更新答案
            results.append(record)

    # 保存结果到输出文件（逐行写入）
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {output_file}")

# 运行主函数
if __name__ == "__main__":
    input_file = "/home/liyanhao/FActScore/data/unlabeled/temp_data.jsonl"  # 输入文件路径
    output_file = "/home/liyanhao/FActScore/data/unlabeled/DeepSeek-R1-Distill-Qwen-7B.jsonl"  # 输出文件路径
    concurrent_limit = 32  # 设置并发数限制
    asyncio.run(process_data(input_file, output_file, concurrent_limit))