import json
from tqdm import tqdm
from swift.tuners import Swift
from swift.llm import (
    PtEngine,
    RequestConfig,
    BaseArguments,
    get_model_tokenizer,
    get_template,
    InferRequest,
)
from utils import *


QWEN2_VL_GENERATE_CONFIG = {
    "top_p": 0.001,
    "top_k": 1,
    "temperature": 0.01,
    "repetition_penalty": 1.0,
    "max_tokens": 2048,
}


def get_infer_engine(adapter_path):
    args = BaseArguments.from_pretrained(adapter_path)
    model, tokenizer = get_model_tokenizer(args.model)
    model = Swift.from_pretrained(model, adapter_path)
    template = get_template(
        args.template, tokenizer, default_system=args.system, max_pixels=301056
    )
    engine = PtEngine.from_model_template(model, template, max_batch_size=1)
    return engine


def get_request_config(adapter_path):
    args = BaseArguments.from_pretrained(adapter_path)
    if "Qwen" in args.model:
        return RequestConfig(**QWEN2_VL_GENERATE_CONFIG)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


def load_request_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    def build_query_history(messages, end_index):
        history = []
        for msg in messages[1:end_index]:
            content = msg["content"] if msg["role"] == "user" else "<history>"
            if msg["role"] not in ["user", "assistant"]:
                raise ValueError(f"Unsupported role: {msg['role']}")
            history.append({"role": msg["role"], "content": content})
        return history

    request_data = []
    for item in tqdm(data, desc="Transforming data"):
        images = item["images"]
        first_msg = item["messages"][0]
        queries = item["messages"][1::2]

        request_item = []
        for i, query in enumerate(queries):
            if i == 0:
                messages = [first_msg, query]
            else:
                history = build_query_history(item["messages"], 2 * i + 1)
                messages = [first_msg] + history + [query]
            request_item.append(InferRequest(messages=messages, images=images))

        request_data.append(request_item)

    return request_data


def run_inference(engine, request_data, request_config):
    results = []
    for request_item in tqdm(request_data, desc="Running inference"):
        histories = []
        for request in request_item:
            if len(histories) >= 1:
                for i, history in enumerate(histories):
                    request.messages[2 * (i + 1)]["content"] = history
            resp_list = engine.infer([request], request_config)
            response = resp_list[0].choices[0].message.content
            histories.append(response)

        # the last dialog contains all histories
        results_item = {
            "messages": request.messages + [{"role": "assistant", "content": response}],
            "images": request.images,
        }
        results.append(results_item)

    return results


def save_metrics_to_file(file_path, metrics):
    with open(file_path, "a") as f:
        for task, metric_data in metrics.items():
            if task in ["cog", "dg", "icg", "ig"]:
                accuracy, precision, recall, f1_score, miou = compute_metrics(
                    metric_data["TP"],
                    metric_data["FP"],
                    metric_data["FN"],
                    metric_data["ious"],
                )
                f.write(
                    f"{task.upper()} Metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Mean IoU: {miou:.4f}\n"
                )
            elif task == "csg":
                miou = (
                    sum(metric_data["ious"]) / len(metric_data["ious"])
                    if metric_data["ious"]
                    else 0
                )
                f.write(f"{task.upper()} Metrics: Mean IoU: {miou:.4f}\n")
            elif task == "cvg":
                accuracy = (
                    metric_data["correct"] / metric_data["total"]
                    if metric_data["total"] > 0
                    else 0
                )
                mean_ed = (
                    sum(metric_data["eds"]) / len(metric_data["eds"])
                    if metric_data["eds"]
                    else 0
                )
                f.write(
                    f"{task.upper()} Metrics: Accuracy: {accuracy:.4f}, Mean Edit Distance: {mean_ed:.4f}\n"
                )


if __name__ == "__main__":

    adapter_path = "../output_mig_40k/v1-20250506-132022/checkpoint-3873"
    request_json_path = "/home/anxiao/Datasets/MIGRANT/mig_2k_val.json"
    result_json_path = "results_mig_2k_val_v1_3.json"

    infer_engine = get_infer_engine(adapter_path)
    request_config = get_request_config(adapter_path)

    request_data = load_request_data(request_json_path)
    results = run_inference(infer_engine, request_data, request_config)
    with open(result_json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    assign_task_type(result_json_path)

    with open(result_json_path, "r") as f:
        data = json.load(f)
    with open(request_json_path, "r") as f:
        request_data = json.load(f)

    metrics = {
        "cog": {"TP": 0, "FP": 0, "FN": 0, "ious": []},
        "csg": {"ious": []},
        "cvg": {"correct": 0, "total": 0, "eds": []},
        "dg": {"TP": 0, "FP": 0, "FN": 0, "ious": []},
        "icg": {"TP": 0, "FP": 0, "FN": 0, "ious": []},
        "ig": {"TP": 0, "FP": 0, "FN": 0, "ious": []},
    }

    eval_functions = {
        "cog": eval_cog_item,
        "csg": eval_csg_item,
        "cvg": eval_cvg_item,
        "dg": eval_dg_item,
        "icg": eval_icg_item,
        "ig": eval_ig_item,
    }

    for item, item_request in tqdm(zip(data, request_data)):
        task_type = item["task_type"]
        if task_type not in eval_functions:
            raise ValueError(f"Unsupported task type: {task_type}")

        eval_func = eval_functions[task_type]
        if task_type == "cog":
            TP, FP, FN, ious = eval_func(item, item_request)
            metrics[task_type]["TP"] += TP
            metrics[task_type]["FP"] += FP
            metrics[task_type]["FN"] += FN
            metrics[task_type]["ious"].extend(ious)
        elif task_type == "csg":
            ious = eval_func(item, item_request)
            metrics[task_type]["ious"].extend(ious)
        elif task_type == "cvg":
            correct, total, eds = eval_func(item, item_request)
            metrics[task_type]["correct"] += correct
            metrics[task_type]["total"] += total
            metrics[task_type]["eds"].extend(eds)
        else:
            TP, FP, FN, ious = eval_func(item, item_request)
            metrics[task_type]["TP"] += TP
            metrics[task_type]["FP"] += FP
            metrics[task_type]["FN"] += FN
            metrics[task_type]["ious"].extend(ious)

    save_metrics_to_file("evaluation_metrics.txt", metrics)
