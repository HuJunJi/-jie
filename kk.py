import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score


# 配置设置
class Config:
    model_name = "bert-base-uncased"
    max_length = 128
    batch_size = 16
    learning_rate = 2e-5
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./glue_results"
    os.makedirs(output_dir, exist_ok=True)


# 数据集类
class GLUEDataset(Dataset):
    def __init__(self, file_path, tokenizer, is_train=True):
        self.tokenizer = tokenizer
        self.is_train = is_train

        # 确保文件存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")

        # 读取数据
        try:
            self.df = pd.read_csv(file_path, sep="\t")
            print(f"成功从 {file_path} 加载 {len(self.df)} 条数据")

            # 检查必要列是否存在
            required_cols = ['sentence1', 'label'] if is_train else ['sentence1']
            for col in required_cols:
                if col not in self.df.columns:
                    raise ValueError(f"缺少必要列: {col}")

        except Exception as e:
            raise ValueError(f"加载数据失败: {str(e)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_a = str(row['sentence1'])
        text_b = str(row['sentence2']) if 'sentence2' in row else None

        inputs = self.tokenizer(
            text_a,
            text_b,
            max_length=Config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        item = {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

        if self.is_train:
            item['label'] = torch.tensor(row['label'], dtype=torch.long)

        return item


# 主函数
def main():
    print("初始化模型和tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # 初始化模型（忽略分类层初始化警告）
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.model_name,
        num_labels=3  # 根据您的任务调整
    )
    model.to(Config.device)
    print(f"模型已加载到 {Config.device}")

    # 加载数据集
    print("\n加载数据集...")
    try:
        train_set = GLUEDataset("AX_train.tsv", tokenizer)
        dev_matched_set = GLUEDataset("AX_dev_matched.tsv", tokenizer)
        dev_mismatched_set = GLUEDataset("AX_dev_mismatched.tsv", tokenizer)
        test_set = GLUEDataset("AX_test.tsv", tokenizer, is_train=False)
        diagnostic_set = GLUEDataset("diagnostic-full.tsv", tokenizer)
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        print("请检查:")
        print("1. 文件是否存在于当前目录")
        print("2. 文件是否包含必需的列(sentence1, label等)")
        print("3. 文件是否为有效的TSV格式")
        return

    # 创建DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=Config.batch_size,
        sampler=RandomSampler(train_set),
        num_workers=0
    )

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * Config.epochs
    )

    # 训练循环
    print("\n开始训练...")
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            optimizer.zero_grad()

            inputs = {
                'input_ids': batch['input_ids'].to(Config.device),
                'attention_mask': batch['attention_mask'].to(Config.device),
                'labels': batch['label'].to(Config.device)
            }

            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        print(f"Epoch {epoch + 1} 平均损失: {total_loss / len(train_loader):.4f}")

    # 保存模型
    model.save_pretrained(Config.output_dir)
    tokenizer.save_pretrained(Config.output_dir)
    print(f"\n模型和tokenizer已保存到 {Config.output_dir}")

    # 在测试集上预测
    test_loader = DataLoader(test_set, batch_size=Config.batch_size)
    predictions = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测"):
            inputs = {
                'input_ids': batch['input_ids'].to(Config.device),
                'attention_mask': batch['attention_mask'].to(Config.device)
            }
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)

    # 保存预测结果
    test_df = pd.read_csv("AX_test.tsv", sep="\t")
    test_df['prediction'] = predictions
    test_df[['index', 'prediction']].to_csv(
        os.path.join(Config.output_dir, "AX.tsv"),
        sep="\t",
        index=False
    )
    print("预测结果已保存")


if __name__ == "__main__":
    main()