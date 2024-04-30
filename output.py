import argparse
import json
from chatpdf import ChatPDF

def load_questions(filename):
    """从 JSON 文件加载问题"""
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_answers(answers, filename):
    """将答案保存到 JSON 文件"""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(answers, file, ensure_ascii=False, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model_name", type=str, default="01-ai/Yi-6B-Chat")
    parser.add_argument("--input_file", type=str, default="question_example.json")
    parser.add_argument("--output_file", type=str, default="answer_example.json")
    args = parser.parse_args()

    # 创建 ChatPDF 实例
    model = ChatPDF(args.gen_model_type, args.gen_model_name, None)

    # 加载问题
    questions = load_questions(args.input_file)

    # 为每个问题生成答案
    answers = []
    for item in questions:
        response = model.predict(item['question'])
        answers.append({
            "id": item['id'],
            "question": item['question'],
            "answer": response
        })

    # 保存答案到文件
    save_answers(answers, args.output_file)
    print(f"答案已保存到 {args.output_file}")

if __name__ == "__main__":
    main()
