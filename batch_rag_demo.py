# -*- coding: utf-8 -*-
"""
这段代码实现了一个用于文本查询和答案生成的系统，使用了一个称为`ChatPDF`的类，该类似乎是为处理文本数据和生成答案设计的。整体流程包括加载数据、处理用户查询、生成答案并评估生成的答案与真实答案之间的一致性。下面逐步解析代码的关键部分和功能：

### 脚本概览
- **导入必要的库**：使用了`argparse`用于解析命令行参数，`json`用于处理JSON格式的数据，等等。
- **定义`get_truth_dict`函数**：这个函数从JSONL格式的文件中读取数据，并构建一个将每个问题映射到其答案的字典。

### 主函数详解
- **参数设置**：通过`argparse.ArgumentParser()`设置并解析了多个命令行参数，例如模型类型、模型名称、输入输出文件等。
- **初始化模型**：创建了一个`ChatPDF`实例，用于处理文本查询并生成答案。
- **加载真实答案**：对于以`.jsonl`结尾的文件，使用`get_truth_dict`函数加载问题和答案到`truth_dict`字典中。
- **处理查询**：
  - 如果未指定查询文件，将使用默认查询列表。
  - 如果指定了查询文件，从该文件中读取查询。
- **批量处理查询**：按照设定的批处理大小，批量处理查询，生成答案，并与真实答案进行比较。
- **结果输出**：将每个查询的输入、生成的输出和真实答案保存到输出文件中。
- **性能统计**：计算处理查询的总时间、处理的查询数和每秒处理的查询数。

### 代码的主要功能
- **批处理**：此脚本以批量方式处理文本查询，适用于处理大量数据时提高效率。
- **性能监测**：脚本计算并输出处理时间和速度，有助于评估模型性能。
- **灵活的输入输出**：通过命令行参数灵活指定输入输出，方便不同情境下的使用。
- **实时反馈**：在控制台实时打印处理的查询和生成的答案，增加了交互性和可追踪性。

### 使用场景
这个脚本可以用于各种需要自动文本查询处理和答案生成的场景，特别是在医疗、法律或任何需要从大量文档中提取信息的领域。结合具体的NLP模型和文本处理算法，可以有效地提供精确的信息检索和内容生成服务。

通过进一步开发和集成，这个脚本的基础功能可以扩展为一个完整的客户支持或信息查询系统，为用户提供即时的问题解答和信息支持。
"""
import argparse
import json
import os
import time

from similarities import BM25Similarity
from tqdm import tqdm

from chatpdf import ChatPDF

pwd_path = os.path.abspath(os.path.dirname(__file__))


def get_truth_dict(jsonl_file_path):
    truth_dict = dict()

    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            input_text = entry.get("question", "")
            output_text = entry.get("answer", "")
            if input_text and output_text:
                truth_dict[input_text] = output_text

    return truth_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model_name", type=str, default="01-ai/Yi-6B-Chat")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--rerank_model_name", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--corpus_files", type=str, default="medical_corpus.jsonl")
    parser.add_argument('--query_file', default="medical_query.txt", type=str, help="query file, one query per line")
    parser.add_argument('--output_file', default='./predictions_result.jsonl', type=str)
    parser.add_argument("--int4", action='store_true', help="use int4 quantization")
    parser.add_argument("--int8", action='store_true', help="use int8 quantization")
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--num_expand_context_chunk", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--test_size", type=int, default=-1)
    args = parser.parse_args()
    print(args)
    sim_model = BM25Similarity()
    model = ChatPDF(
        similarity_model=sim_model,
        generate_model_type=args.gen_model_type,
        generate_model_name_or_path=args.gen_model_name,
        lora_model_name_or_path=args.lora_model,
        corpus_files=args.corpus_files.split(','),
        device=args.device,
        int4=args.int4,
        int8=args.int8,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rerank_model_name_or_path=args.rerank_model_name,
        num_expand_context_chunk=args.num_expand_context_chunk,
    )
    print(f"chatpdf model: {model}")

    truth_dict = dict()
    for i in args.corpus_files.split(','):
        if i.endswith('.jsonl'):
            tmp_truth_dict = get_truth_dict(i)
            truth_dict.update(tmp_truth_dict)
    print(f"truth_dict size: {len(truth_dict)}")
    # test data
    if args.query_file is None:
        examples = ["肛门病变可能是什么疾病的症状?", "膺窗穴的定位是什么?"]
    else:
        with open(args.query_file, 'r', encoding='utf-8') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    if args.test_size > 0:
        examples = examples[:args.test_size]
    print("Start inference.")
    t1 = time.time()
    counts = 0
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    eval_batch_size = args.eval_batch_size
    for batch in tqdm(
            [
                examples[i: i + eval_batch_size]
                for i in range(0, len(examples), eval_batch_size)
            ],
            desc="Generating outputs",
    ):
        results = []
        for example in batch:
            response, reference_results = model.predict(example)
            truth = truth_dict.get(example, '')
            print(f"===")
            print(f"Input: {example}")
            print(f"Reference: {reference_results}")
            print(f"Output: {response}")
            print(f"Truth: {truth}\n")
            results.append({"Input": example, "Output": response, "Truth": truth})
            counts += 1
        with open(args.output_file, 'a', encoding='utf-8') as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    t2 = time.time()
    print(f"Saved to {args.output_file}, Time cost: {t2 - t1:.2f}s, size: {counts}, "
          f"speed: {counts / (t2 - t1):.2f} examples/s")
