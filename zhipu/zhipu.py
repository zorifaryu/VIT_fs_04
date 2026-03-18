import pandas as pd
from zhipuai import ZhipuAI
import time

# ================= 配置区 =================
# 填入你刚才创建的正式 API Key
ZHIPU_API_KEY = "af44c0cd4e3147008147ab4386e74d91.QEZBNAMQEiCFQ3Xf "

client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 文件路径配置
PATH_EXCEL = 'cleaned_data.csv'  # 对应你 Mac 项目里的文件名 [cite: 7, 14]
NAME_FINAL = "广东招聘摘要结果"


# ================= 核心函数 =================
def get_summary(text):
    # 严格遵循原代码的 Prompt 逻辑 [cite: 2, 6]
    prompt = ("请给出以下关于工作要求信息的摘要，需要只保留关于该工作岗位的要求，"
              "字数尽量控制在20字左右，不保留公司电话、工作时间、地点、公司待遇及薪资等无关信息，"
              "并且摘要内容不能换行。以下是工作岗位要求：\n " + str(text))

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",  # 免费额度最高的模型
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,  # 对应原代码参数
            top_p=0.6,  # 对应原代码参数
            max_tokens=50  # 对应原代码参数
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"处理出错: {e}"


# ================= 执行区 =================
def main():
    print("正在读取数据...")
    df = pd.read_excel(PATH_EXCEL)  # [cite: 7]

    # 建议先测 5 条，成功后再改大
    start = 10001
    end = 20000

    results = []
    for i in range(start,end):
        # 对应你 Excel 中的列名，请确保列名正确 [cite: 8, 14]
        content = df.loc[i, '职位描述_clean']
        city = df.loc[i, '工作城市']

        print(f"正在处理第 {i + 1} 条 ({city})...")
        res = get_summary(content)
        print(f"摘要: {res}")

        results.append({'序号': i + 1, '城市': city, '摘要': res})
        time.sleep(0.5)  # 智谱速度很快，不用等很久

    # 保存
    pd.DataFrame(results).to_excel(NAME_FINAL + ".xlsx", index=False)  # [cite: 12]
    print(f"已保存至 {NAME_FINAL}.xlsx")


if __name__ == "__main__":
    main()
