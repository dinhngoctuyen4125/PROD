from tqdm import tqdm
import random
import numpy as np
from functools import partial

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser,Seq2SeqTrainingArguments

import wandb
wandb.init(project="unlearn_code", name="PROD")


def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# def calculate_loss(model_disprefered_logits, ground_truth_distribution):
    
#     model_disprefered_distribution = F.softmax(model_disprefered_logits, dim=-1)
        
#     model_disprefered_distribution = model_disprefered_distribution[..., :-1, :].contiguous()

#     log_probs = torch.log(model_disprefered_distribution + 1e-5)

#     cross_entropy_loss = -torch.sum(ground_truth_distribution * log_probs, dim=-1)
#     mean_cross_entropy_loss = torch.mean(cross_entropy_loss)
    
#     return mean_cross_entropy_loss

def calculate_loss(model_logits, ground_truth_distribution, loss_mask):
    model_dist = F.softmax(model_logits[:, :-1, :], dim=-1)
    log_probs = torch.log(model_dist + 1e-5)
    
    # Tính Cross Entropy (Shape: B x L-1)
    ce_loss = -torch.sum(ground_truth_distribution * log_probs, dim=-1)
    
    shifted_loss_mask = loss_mask[:, 1:]
    valid_tokens = shifted_loss_mask.sum()
    
    # Chỉ lấy trung bình loss trên các token của phần cần quên
    if valid_tokens > 0:
        mean_ce_loss = (ce_loss * shifted_loss_mask).sum() / valid_tokens
    else:
        mean_ce_loss = ce_loss.sum() * 0.0 # Tránh lỗi NaN nếu batch không có token nào hợp lệ
        
    return mean_ce_loss


def top_p_filtering(logits, top_p=0.9, filter_value=0.0, N=1, max_N=10, need_softmax=True):

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    if max_N is None:
        max_N = probs.size(-1)
    max_N_mask = torch.arange(sorted_probs.size(-1), device=logits.device) >= max_N
    sorted_indices_to_remove |= max_N_mask

    remove_mask = torch.zeros_like(probs, dtype=torch.bool)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    filtered_logits = logits.masked_fill(remove_mask, filter_value)
    
    return filtered_logits
    

# def get_output_distribution(logits, labels, top_p=0.8, alpha=0.0, temperature=0.8, N=1, max_N=10):
#     probs = F.softmax(logits, dim=-1)

#     with torch.no_grad():
#         labels = labels[..., 1:]

#         copied_logits = logits[..., :-1, :].clone()

#         mask_start_pos = 1
#         mask_start = torch.zeros_like(labels, dtype=torch.bool) 
#         mask_start[:, mask_start_pos:] = 1 
#         labels = labels.long()
#         mask = F.one_hot(labels, num_classes=copied_logits.size(-1)) & mask_start.unsqueeze(-1)
#         copied_logits = copied_logits.masked_fill(mask.bool(), -float('inf'))

#         filtered_logit = top_p_filtering(copied_logits, top_p=top_p, N=N, max_N=max_N, filter_value=-float('inf'))

#         if temperature is None:
#             scaled_logit = filtered_logit
#         else:
#             scaled_logit = filtered_logit / temperature

#         ground_truth_probs = F.softmax(scaled_logit, dim=-1)

#         one_hot = F.one_hot(labels, num_classes=probs.size(-1)).bool()
#         ground_truth_probs = torch.where(one_hot, -alpha*probs[..., :-1, :], ground_truth_probs)

#     return probs, ground_truth_probs

def get_output_distribution(logits, input_ids, loss_mask, top_p=0.8, alpha=0.0, temperature=0.8, N=1, max_N=10):
    probs = F.softmax(logits, dim=-1)

    with torch.no_grad():
        # Dịch chuyển nhãn và mask để căn chỉnh với logits (dự đoán token tiếp theo)
        labels = input_ids[:, 1:].long()
        shifted_loss_mask = loss_mask[:, 1:].bool()
        copied_logits = logits[:, :-1, :].clone()

        # Tạo mask cho token mục tiêu cần triệt tiêu (chỉ triệt tiêu phần target, không triệt tiêu prompt)
        target_one_hot = F.one_hot(labels, num_classes=copied_logits.size(-1)).bool()
        valid_suppression_mask = target_one_hot & shifted_loss_mask.unsqueeze(-1)
        
        # Gán logit của token mục tiêu bằng -inf
        copied_logits.masked_fill_(valid_suppression_mask, -float('inf'))

        filtered_logit = top_p_filtering(copied_logits, top_p=top_p, N=N, max_N=max_N, filter_value=-float('inf'))

        if temperature is not None:
            filtered_logit = filtered_logit / temperature

        ground_truth_probs = F.softmax(filtered_logit, dim=-1)

        # Kỹ thuật alpha-suppression chỉ áp dụng trên các token thuộc phần cần quên
        ground_truth_probs = torch.where(valid_suppression_mask, -alpha * probs[:, :-1, :], ground_truth_probs)

    return probs, ground_truth_probs


# def collate_fn(batch, tokenizer, max_length, device):
#     prompts = [item['input'] for item in batch]
#     rejected_responses = [item['deprecated_api'] for item in batch]

#     prompt_ids = tokenizer(prompts, padding=True, return_tensors="pt", max_length=max_length, truncation=True, add_special_tokens=True)['input_ids'].to(device)
    
#     disprefered_ids = tokenizer(rejected_responses, padding=True, return_tensors="pt", max_length=max_length, truncation=True, add_special_tokens=False)['input_ids'].to(device)

#     prompt_disprefered_ids = torch.cat([prompt_ids, disprefered_ids], dim=-1)
#     prompt_disprefered_mask = torch.ones_like(prompt_disprefered_ids)

#     return {'prompt_disprefered_ids': prompt_disprefered_ids,
#             'prompt_disprefered_mask': prompt_disprefered_mask}

def collate_fn(batch, tokenizer, max_length, device):
    input_ids_list = []
    loss_mask_list = []

    for item in batch:
        prompt = item['input']
        target = item['deprecated_api']

        # Tokenize không có padding để lấy chính xác số lượng token
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        target_ids = tokenizer.encode(target, add_special_tokens=False)

        full_ids = prompt_ids + target_ids
        
        # Mask: 0 cho prompt (không tính loss), 1 cho target (phần cần quên)
        mask = [0] * len(prompt_ids) + [1] * len(target_ids)

        # Truncate nếu vượt quá max_length (Cắt bớt phần đầu của prompt để giữ target)
        if len(full_ids) > max_length:
            excess = len(full_ids) - max_length
            full_ids = full_ids[excess:]
            mask = mask[excess:]

        input_ids_list.append(torch.tensor(full_ids))
        loss_mask_list.append(torch.tensor(mask))

    # Pad toàn bộ batch sau khi đã nối chuỗi
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    padded_loss_mask = torch.nn.utils.rnn.pad_sequence(loss_mask_list, batch_first=True, padding_value=0)
    attention_mask = (padded_input_ids != pad_token_id).long()

    return {
        'input_ids': padded_input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'loss_mask': padded_loss_mask.to(device)
    }


# def train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=1, gradient_accumulation_steps=1, top_p=0.8, temperature=0.8, N=1, max_N=10, alpha=0.0):
#     model.train()

#     for epoch in range(int(epochs)):
#         print(f"Epoch {epoch + 1}/{epochs}")
#         optimizer.zero_grad()
        
#         for step, batch in enumerate(tqdm(train_dataloader)):
#             prompt_disprefered_ids = batch['prompt_disprefered_ids']
#             prompt_disprefered_mask = batch['prompt_disprefered_mask']

#             with torch.no_grad():
#                 _, ground_truth_distribution = get_output_distribution(ref_model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits, 
#                                                                        prompt_disprefered_ids, 
#                                                                        top_p=top_p, 
#                                                                        alpha=alpha,
#                                                                        temperature=temperature, 
#                                                                        N=N, 
#                                                                        max_N=max_N)

#             model_disprefered_logits = model(prompt_disprefered_ids, attention_mask=prompt_disprefered_mask).logits

#             loss = calculate_loss(model_disprefered_logits, ground_truth_distribution)
            
#             loss = loss / gradient_accumulation_steps
#             loss.backward()

#             if (step + 1) % gradient_accumulation_steps == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()

#         if len(train_dataloader) % gradient_accumulation_steps != 0:
#             optimizer.step()
#             optimizer.zero_grad()

#         optimizer.zero_grad()

#         print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")
#         wandb.log({'epoch loss': loss.item()})

#         # every epoch, save the model
#         output_dir = wandb.config.output_dir + "/" + f"PROD_epoch{epoch}_lr{wandb.config.learning_rate}"
#         model.save_pretrained(output_dir)
#         tokenizer.save_pretrained(output_dir)

def train(model, ref_model, tokenizer, optimizer, train_dataloader, epochs=1, gradient_accumulation_steps=1, top_p=0.8, temperature=0.8, N=1, max_N=10, alpha=0.0):
    model.train()

    for epoch in range(int(epochs)):
        print(f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            loss_mask = batch['loss_mask']

            with torch.no_grad():
                # Lấy output distribution của reference model
                ref_logits = ref_model(input_ids, attention_mask=attention_mask).logits
                _, ground_truth_distribution = get_output_distribution(
                    ref_logits, input_ids, loss_mask, 
                    top_p=top_p, alpha=alpha, temperature=temperature, N=N, max_N=max_N
                )

            # Tính logits của model hiện tại
            model_logits = model(input_ids, attention_mask=attention_mask).logits

            loss = calculate_loss(model_logits, ground_truth_distribution, loss_mask)
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                
                # Log wandb theo step để dễ theo dõi sự hội tụ
                wandb.log({'step_loss': loss.item() * gradient_accumulation_steps})

        print(f"Epoch [{epoch+1}/{epochs}], Loss cuối: {loss.item() * gradient_accumulation_steps}")

        # Lưu checkpoint sau mỗi epoch
        output_dir = wandb.config.output_dir + "/" + f"PROD_epoch{epoch}_lr{wandb.config.learning_rate}"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


@dataclass
class CustomArguments:
    model_name: str = field(default='codellama/CodeLlama-7b-hf')
    model_path: str = field(default=None)
    last_checkpoint: str = field(default=None)
    train_data_path: str = field(default='data/forget_data/merged_deprecated_apis.json')
    max_seq_length: int = field(default=1024)
    lora_rank: int = field(default=16)
    top_p: float = field(default=0.8)
    temperature: float = field(default=None)
    N: int = field(default=1)
    max_N: int = field(default=None)
    alpha: float = field(default=0.0)


def main():
    hf_parser = HfArgumentParser((Seq2SeqTrainingArguments, CustomArguments))
    training_args, custom_args = hf_parser.parse_args_into_dataclasses()
    print(training_args)
    print(custom_args)

    wandb.config.update(training_args)
    wandb.config.update(custom_args)

    if custom_args.model_path is not None:
        model_path = custom_args.model_path
    else:
        model_path = custom_args.model_name

    num_gpus = torch.cuda.device_count()

    total_memory = []
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory.append(int(props.total_memory / (1024 ** 3)))

    model_max_memory = {}
    for i in range(num_gpus):
        if i < num_gpus - 1:
            model_max_memory[i] = f'{total_memory[i]}GB'
        else:
            model_max_memory[i] = '0GB'

    ref_model_max_memory = {}
    for i in range(num_gpus):
        if i < num_gpus - 1:
            ref_model_max_memory[i] = '0GB'
        else:
            ref_model_max_memory[i] = f'{total_memory[i]}GB'


    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ------------------
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        torch_dtype=torch.bfloat16
    )
    ref_model.config.use_cache = False
    ref_model.config.pretraining_tp = 1
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(custom_args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from accelerate import Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    model = accelerator.prepare(model)
    ref_model = accelerator.prepare(ref_model)
    # -----------

    # use parameters from training_args to set up optimizer
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, eps=training_args.adam_epsilon, weight_decay=training_args.weight_decay, betas=(training_args.adam_beta1, training_args.adam_beta2))

    dataset = load_dataset('json', data_files=custom_args.train_data_path)['train']
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True, collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=custom_args.max_seq_length, device=device))

    train(model, 
            ref_model, 
            tokenizer, 
            optimizer, 
            train_dataloader,
            epochs=training_args.num_train_epochs,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            top_p=custom_args.top_p,
            alpha=custom_args.alpha,
            temperature=custom_args.temperature,
            N=custom_args.N,
            max_N=custom_args.max_N)


if __name__ == "__main__":
    seed_everything(42)
    main()
