import pandas as pd
def transform_cora_content_to_csv():
    # === 1. 读取 cora.content ===
    # cora.content 是以空格或制表符分隔，因此我们使用 \t 或空白自动分隔
    df = pd.read_csv("cora.content", sep=r"\s+", header=None)

    # === 2. 构造表头 ===
    # cora.content 格式为：
    # <paper_id> <1433维特征> <label>
    num_columns = df.shape[1]
    num_features = num_columns - 2  # 除去 id 和 label

    columns = ["id"] + [f"f{i}" for i in range(1, num_features + 1)] + ["label"]
    df.columns = columns

    # === 3. 导出为 CSV ===
    df.to_csv("cora.csv", index=False)

    print("已成功生成 cora.csv")