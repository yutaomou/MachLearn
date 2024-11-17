from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer
classifier = pipeline("sentiment-analysis") # 加载预训练模型(情感分析模型)
# print(classifier("We are a shit")) # 预测结果
results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}") #对于多个句子的情感分析