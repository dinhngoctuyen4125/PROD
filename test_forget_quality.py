import os
import argparse
import json
import logging
import pprint
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu

from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer

set_seed(42)
MAX_GENERATION_LENGTH = 128


# def sample_code_from_llm(args, prompt, model, tokenizer):
#     completions = []

#     if tokenizer.bos_token_id:
#         input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False, verbose=False) 
#         input_ids = torch.tensor([input_ids]).to("cuda:0")
#         eos_token = tokenizer.eos_token_id        
#     else:
#         input_ids = tokenizer.encode(prompt, add_special_tokens=False, verbose=False) 
#         input_ids = torch.tensor([input_ids]).to("cuda:0")
#         eos_token = tokenizer.eos_token_id

#     num_return_sequences = args.acctual_num_samples
#     if args.temperature == 0.0:
#         args.num_samples = 1
#         num_return_sequences = 1

#     model.eval()

#     for i in range(int(args.num_samples/num_return_sequences)):
#         try:
#             if args.temperature > 0:
#                 tokens = model.generate(
#                     input_ids,
#                     do_sample=True,
#                     num_return_sequences=num_return_sequences,
#                     max_new_tokens=MAX_GENERATION_LENGTH,
#                     temperature=args.temperature,
#                     use_cache=True,
#                     top_k=args.topk,
#                     top_p=args.topp,
#                     eos_token_id=eos_token,
#                 )
#             else:
#                 tokens = model.generate(
#                         input_ids,
#                         num_return_sequences=1,
#                         max_new_tokens=MAX_GENERATION_LENGTH,
#                         use_cache=True,
#                         do_sample=False,
#                         eos_token_id=eos_token,
#                     )

#             for i in tokens:
#                 i = i[input_ids.shape[1]:]
#                 text = tokenizer.decode(i, skip_special_tokens=False)
#                 completions.append(text)
                
#         except RuntimeError as e:
#             logging.error(f"Could not sample from model: {e}")

#     return completions

def sample_code_from_llm(args, prompt, model, tokenizer):
    completions = []
    
    # Mã hóa prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, verbose=False) 
    if tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
        
    input_tensor = torch.tensor([input_ids]).to("cuda:0")
    
    # XỬ LÝ SEQUENCE LENGTH: Truncate từ bên trái nếu prompt quá dài (ví dụ: giới hạn 2048 token)
    MAX_PROMPT_LENGTH = 2048
    if input_tensor.shape[1] > MAX_PROMPT_LENGTH:
        input_tensor = input_tensor[:, -MAX_PROMPT_LENGTH:]
        
    eos_token = tokenizer.eos_token_id
    num_return_sequences = args.acctual_num_samples
    if args.temperature == 0.0:
        args.num_samples = 1
        num_return_sequences = 1

    model.eval()

    for i in range(int(args.num_samples/num_return_sequences)):
        try:
            if args.temperature > 0:
                tokens = model.generate(
                    input_tensor,
                    do_sample=True,
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=MAX_GENERATION_LENGTH,
                    temperature=args.temperature,
                    use_cache=True,
                    top_k=args.topk,
                    top_p=args.topp,
                    eos_token_id=eos_token,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                tokens = model.generate(
                    input_tensor,
                    num_return_sequences=1,
                    max_new_tokens=MAX_GENERATION_LENGTH,
                    use_cache=True,
                    do_sample=False,
                    eos_token_id=eos_token,
                    pad_token_id=tokenizer.eos_token_id
                )

            for idx in tokens:
                # Chỉ lấy phần code được sinh ra, bỏ qua phần prompt
                generated_tokens = idx[input_tensor.shape[1]:]
                text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                completions.append(text)
                
        except RuntimeError as e:
            logging.error(f"Could not sample from model: {e}")

    return completions

def load_model_tokenizer(args, model_name, model_path):

    if model_path:
        model_path = model_path
    else:
        model_path = model_name
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16,
        device_map={"": 0}
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    generate_code_fn = lambda args, prompt: sample_code_from_llm(
        args, prompt, model, tokenizer
    )

    return generate_code_fn, tokenizer

# def generate_code_for_tasks(args, except_tasks, save_file):

#     # open save file
#     f = open(save_file, "a")
#     summary_f = open(save_file.replace(".jsonl", "_summary.txt"), "a")

#     generate_code_fn, tokenizer = load_model_tokenizer(args, args.model_name, args.model_path)

#     # load dataset
#     dataset = load_dataset('json', data_files=args.dataset)['train']

#     AVG_BLEU = 0
#     for i in tqdm(range(len(dataset))):
#         # SỬA LỖI DATASET: Tạo task_id ảo từ số thứ tự i (chuyển thành chuỗi)
#         task_id = str(i)

#         if (task_id in except_tasks):
#             continue

#         # SỬA LỖI DATASET: Lấy prompt từ cột 'input' và ground_truth từ 'deprecated_api'
#         prompt = dataset[i]["input"]

#         if prompt == "":
#             parts = ground_truth.split(" ")
#             length = len(parts)
#             prompt = " ".join(parts[:length//2])
#             ground_truth = " " + " ".join(parts[length//2:])

#         # generate code and write to file
#         BLEU = 0
#         for completion in generate_code_fn(args, prompt):
#             output ={
#                     "task_id": task_id,
#                     "prompt": prompt,
#                     "completion": completion,
#                 }
#             BLEU += sentence_bleu([tokenizer.tokenize(ground_truth)], tokenizer.tokenize(completion))
#             f.write(json.dumps(output) + "\n")
#             f.flush()
            
#         # SỬA LỖI 2: Đẩy biến AVG_BLEU thụt lề vào trong đúng vòng lặp của từng task
#         AVG_BLEU += BLEU / args.num_samples

#     AVG_BLEU /= len(dataset)
#     print(f"Average BLEU: {AVG_BLEU}")
#     summary_f.write(f"Average BLEU: {AVG_BLEU}\n")

#     f.close()
#     summary_f.close()

def generate_code_for_tasks(args, except_tasks, save_file):
    f = open(save_file, "a")
    summary_f = open(save_file.replace(".jsonl", "_summary.txt"), "a")

    generate_code_fn, tokenizer = load_model_tokenizer(args, args.model_name, args.model_path)
    dataset = load_dataset('json', data_files=args.dataset)['train']

    total_exact_match = 0
    total_bleu = 0
    processed_count = 0

    for i in tqdm(range(len(dataset))):
        task_id = str(i)
        if task_id in except_tasks:
            continue

        prompt = dataset[i]["input"]
        ground_truth = dataset[i]["deprecated_api"] # FIX LỖI THIẾU BIẾN

        if prompt == "":
            parts = ground_truth.split(" ")
            length = len(parts)
            prompt = " ".join(parts[:length//2])
            ground_truth = " " + " ".join(parts[length//2:])

        task_bleu = 0
        task_exact_match = 0
        
        for completion in generate_code_fn(args, prompt):
            output = {
                "task_id": task_id,
                "prompt": prompt,
                "completion": completion,
                "ground_truth": ground_truth
            }
            
            # 1. Đánh giá bằng Exact Match (Đúng theo bài báo PROD cho Deprecated API)
            # Nếu API cũ xuất hiện trong đoạn code sinh ra -> mô hình CHƯA quên thành công
            if ground_truth.strip() in completion:
                task_exact_match += 1
                
            # 2. Vẫn tính BLEU để tham khảo thêm
            task_bleu += sentence_bleu([tokenizer.tokenize(ground_truth)], tokenizer.tokenize(completion))
            
            f.write(json.dumps(output) + "\n")
            f.flush()
            
        total_exact_match += task_exact_match / args.num_samples
        total_bleu += task_bleu / args.num_samples
        processed_count += 1

    if processed_count > 0:
        avg_exact_match_rate = total_exact_match / processed_count
        # Forget Quality = 1 - Tỉ lệ mô hình vẫn sinh ra API cũ
        forget_quality = 1.0 - avg_exact_match_rate 
        avg_bleu = total_bleu / processed_count
        
        print(f"Forget Quality (Exact Match): {forget_quality:.4f}")
        print(f"Average BLEU: {avg_bleu:.4f}")
        
        summary_f.write(f"Processed Tasks: {processed_count}\n")
        summary_f.write(f"Forget Quality (Exact Match): {forget_quality:.4f}\n")
        summary_f.write(f"Average BLEU: {avg_bleu:.4f}\n")
    else:
        print("Không có task nào được xử lý (có thể đã resume xong hết).")

    f.close()
    summary_f.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="CodeLlama-7b-hf")
    parser.add_argument("--model_path", default=None, help="Directory where a pre-trained LLM or fine-tuned LLM is saved. If None, will load from huggingface cache.",)
    parser.add_argument("--dataset", type=str)    
    parser.add_argument("--num-samples", default=1, type=int)
    parser.add_argument("--acctual-num-samples", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--topp", default=None, type=float)
    parser.add_argument("--topk", default=None, type=int)
    parser.add_argument("--few-shot", default=0, type=int)
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--output-file-suffix", type=str, default="")
    args = parser.parse_args()
    return args


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_name = args.model_name.split("/")[-1]
    dataset = args.dataset.split("/")[-1]
    save_file = os.path.join(
        args.output_dir,
        f"{dataset}_{model_name}_temp{args.temperature}_topp{args.topp}_topk{args.topk}_samples{args.num_samples}_{args.few_shot}shot_{args.output_file_suffix}.jsonl",
    )
    
    except_tasks = []
    if os.path.exists(save_file):
        print(f"File {save_file} already exists in {args.output_dir}.")
        lines = open(save_file).readlines()
        for line in lines:
            # Ép kiểu về string để khớp với task_id ảo tạo ở trên
            task_id = str(json.loads(line)["task_id"])
            if task_id not in except_tasks:
                except_tasks.append(task_id)
    
    # SỬA LỖI 3: Xóa lệnh if cản trở việc chạy tiếp (resume), gọi hàm luôn
    generate_code_for_tasks(args, except_tasks, save_file)


if __name__ == "__main__":
    main(parse_args())