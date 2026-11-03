import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

EVENT_TYPES = ["无事件", "交通事故", "拥堵", "施工", "抛洒物", "异常停车", "行人闯入", "清障", "收费站", "其他"]
INCIDENT_LEVELS = [0, 1, 2, 3]
SCENE_REGIONS = ["左侧", "中间", "右侧", "上方", "下方", "近距离", "远距离", "未知"]
ROAD_LOCATIONS = ["应急车道", "行车道", "超车道", "正常路段", "匝道", "收费站", "未知"]
LANES = ["应急车道", "行车道", "超车道", "无", "未知"]
WEATHERS = ["晴", "阴", "雨", "雪", "雾", "霾", "未知"]
VISIBILITIES = [1, 2, 3]
LIGHTINGS = ["白天", "夜间", "黄昏", "人工照明", "未知"]
IMPACTS = [0, 1, 2, 3]


def safe_div(a, b):
    return a / b if b else 0.0


def normalize_event_type(value):
    text = str(value or "")
    mapping = [
        (["无事件", "正常情况", "正常", "无事故"], "无事件"),
        (["交通事故", "事故", "碰撞", "追尾", "刮擦", "侧翻", "相撞"], "交通事故"),
        (["拥堵", "交通拥堵", "congestion"], "拥堵"),
        (["施工", "道路施工", "construction"], "施工"),
        (["抛洒物", "抛撒物", "debris"], "抛洒物"),
        (["异常停车", "停车", "违停", "故障停车", "应急车道停车", "行车道停车"], "异常停车"),
        (["行人", "行人闯入", "pedestrian"], "行人闯入"),
        (["清障", "清理", "clearance"], "清障"),
        (["收费站", "toll"], "收费站"),
    ]
    for keys, label in mapping:
        if any(key in text for key in keys):
            return label
    return "其他"


def normalize_level(value):
    if isinstance(value, bool):
        return "未知"
    if isinstance(value, (int, float)):
        number = int(value)
        return number if number in (0, 1, 2, 3) else "未知"
    text = str(value or "").strip().lower()
    if text in ("0", "无", "无事件", "正常", "none"):
        return 0
    if text in ("1", "轻微", "轻", "minor", "low"):
        return 1
    if text in ("2", "一般", "中等", "中", "moderate", "medium"):
        return 2
    if text in ("3", "严重", "重", "高", "severe", "critical", "high"):
        return 3
    return "未知"


def normalize_scene_region(value):
    text = str(value or "")
    mapping = [
        (["左侧", "左边", "left"], "左侧"),
        (["右侧", "右边", "right"], "右侧"),
        (["中间", "中央", "center", "middle"], "中间"),
        (["上方", "上边", "top"], "上方"),
        (["下方", "下边", "bottom"], "下方"),
        (["近距离", "较近", "近处", "near"], "近距离"),
        (["远距离", "较远", "远处", "far"], "远距离"),
    ]
    for keys, label in mapping:
        if any(key in text for key in keys):
            return label
    return "未知"


def normalize_road_location(value):
    if isinstance(value, str):
        items = re.split(r"[，,；;、/|]", value)
    elif isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value]
    else:
        items = []
    found = []
    for item in items:
        text = item.strip()
        if not text:
            continue
        if any(key in text for key in ["应急车道", "硬路肩", "路肩", "emergency"]):
            found.append("应急车道")
        elif any(key in text for key in ["超车道", "快车道", "overtaking"]):
            found.append("超车道")
        elif any(key in text for key in ["行车道", "主车道", "正常车道", "travel lane"]):
            found.append("行车道")
        elif "正常路段" in text:
            found.append("正常路段")
        elif "匝道" in text:
            found.append("匝道")
        elif "收费站" in text:
            found.append("收费站")
        elif any(key in text for key in ["无", "不影响", "unknown", "未知"]):
            found.append("未知")
    if "应急车道" in found or "行车道" in found or "超车道" in found:
        found = [x for x in found if x not in ("未知",)]
    return sorted(set(found)) or ["未知"]


def normalize_affected_lanes(value):
    if isinstance(value, str):
        items = re.split(r"[，,；;、/|]", value)
    elif isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value]
    else:
        items = []
    found = []
    for item in items:
        text = item.strip()
        if not text:
            continue
        if any(key in text for key in ["应急车道", "硬路肩", "路肩", "emergency"]):
            found.append("应急车道")
        elif any(key in text for key in ["超车道", "快车道", "overtaking"]):
            found.append("超车道")
        elif any(key in text for key in ["行车道", "主车道", "正常车道", "travel lane"]):
            found.append("行车道")
        elif any(key in text for key in ["无", "未影响", "none"]):
            found.append("无")
        elif any(key in text for key in ["未知", "unknown"]):
            found.append("未知")
    if any(x in found for x in ("应急车道", "行车道", "超车道")):
        found = [x for x in found if x != "无"]
        found = [x for x in found if x != "未知"]
    return sorted(set(found)) or ["未知"]


def normalize_weather(value):
    text = str(value or "")
    mapping = [
        (["雾", "fog"], "雾"),
        (["雪", "snow"], "雪"),
        (["雨", "rain"], "雨"),
        (["霾", "haze"], "霾"),
        (["阴", "overcast", "cloudy"], "阴"),
        (["晴", "clear", "sunny", "正常"], "晴"),
    ]
    for keys, label in mapping:
        if any(key in text for key in keys):
            return label
    return "未知"


def normalize_visibility(value):
    if isinstance(value, bool):
        return "未知"
    if isinstance(value, (int, float)):
        number = int(value)
        return number if number in (1, 2, 3) else "未知"
    text = str(value or "").strip().lower()
    if text in ("3", "高", "良好", "好", "high"):
        return 3
    if text in ("2", "中", "一般", "medium", "moderate"):
        return 2
    if text in ("1", "低", "差", "不足", "low"):
        return 1
    return "未知"


def normalize_lighting(value):
    text = str(value or "")
    if any(key in text for key in ["夜间", "夜晚", "黑夜", "night"]):
        return "夜间"
    if any(key in text for key in ["黄昏", "傍晚", "dusk"]):
        return "黄昏"
    if any(key in text for key in ["白天", "日间", "day"]):
        return "白天"
    if any(key in text for key in ["人工照明", "隧道", "灯光", "artificial"]):
        return "人工照明"
    return "未知"


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "是", "yes", "y"):
        return True
    if text in ("false", "0", "否", "no", "n"):
        return False
    return None


def normalize_count(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def accuracy(pairs):
    if not pairs:
        return None
    return sum(1 for gold, pred in pairs if gold == pred) / len(pairs)


def macro_f1(pairs):
    if not pairs:
        return None
    labels = sorted(set(g for g, _ in pairs) | set(p for _, p in pairs), key=str)
    scores = []
    per_class = {}
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for g, _ in pairs if g == label),
        }
        scores.append(f1)
    return {"macro_f1": sum(scores) / len(scores), "per_class": per_class}


def qwk(pairs, labels):
    if not pairs:
        return None
    n = len(labels)
    index = {label: i for i, label in enumerate(labels)}
    conf = [[0] * n for _ in range(n)]
    hist_gold = [0] * n
    hist_pred = [0] * n
    for gold, pred in pairs:
        if gold not in index or pred not in index:
            continue
        gi, pi = index[gold], index[pred]
        conf[gi][pi] += 1
        hist_gold[gi] += 1
        hist_pred[pi] += 1
    total = sum(hist_gold)
    if total == 0:
        return None
    numerator = sum((i - j) ** 2 * conf[i][j] for i in range(n) for j in range(n))
    denominator = sum((i - j) ** 2 * hist_gold[i] * hist_pred[j] / total for i in range(n) for j in range(n))
    return 1 - numerator / denominator if denominator else 0.0


def multilabel_metrics(gold_sets, pred_sets):
    if not gold_sets:
        return None
    labels = sorted(set().union(*gold_sets, *pred_sets))
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    fp = sum(len(p - g) for g, p in zip(gold_sets, pred_sets))
    fn = sum(len(g - p) for g, p in zip(gold_sets, pred_sets))
    micro = safe_div(2 * tp, 2 * tp + fp + fn)
    scores = []
    per_class = {}
    for label in labels:
        tp_l = sum(1 for g, p in zip(gold_sets, pred_sets) if label in g and label in p)
        fp_l = sum(1 for g, p in zip(gold_sets, pred_sets) if label not in g and label in p)
        fn_l = sum(1 for g, p in zip(gold_sets, pred_sets) if label in g and label not in p)
        precision = safe_div(tp_l, tp_l + fp_l)
        recall = safe_div(tp_l, tp_l + fn_l)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        scores.append(f1)
    macro = sum(scores) / len(scores) if scores else 0.0
    exact = sum(1 for g, p in zip(gold_sets, pred_sets) if g == p) / len(gold_sets)
    return {
        "n": len(gold_sets),
        "micro_f1": micro,
        "macro_f1": macro,
        "exact_match": exact,
        "per_class": per_class,
    }


def char_bigrams(text):
    text = re.sub(r"\s+", "", text or "")
    if len(text) < 2:
        return Counter(text)
    return Counter(text[i:i + 2] for i in range(len(text) - 1))


def char_bigram_f1(reference, prediction):
    ref = char_bigrams(reference)
    pred = char_bigrams(prediction)
    tp = sum((ref & pred).values())
    fp = sum(pred.values()) - tp
    fn = sum(ref.values()) - tp
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return safe_div(2 * precision * recall, precision + recall)


def char_rouge_l(reference, prediction):
    ref = re.sub(r"\s+", "", reference or "")
    pred = re.sub(r"\s+", "", prediction or "")
    if not ref or not pred:
        return 0.0
    m, n = len(ref), len(pred)
    previous = [0] * (n + 1)
    for i in range(1, m + 1):
        current = [0] * (n + 1)
        ref_char = ref[i - 1]
        for j in range(1, n + 1):
            if ref_char == pred[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    lcs = previous[n]
    precision = lcs / n
    recall = lcs / m
    return safe_div(2 * precision * recall, precision + recall)


def composite_report(pairs):
    result = {
        "n": len(pairs),
        "accuracy": accuracy(pairs),
        "macro_f1": None,
    }
    if pairs:
        result["macro_f1"] = macro_f1(pairs)["macro_f1"]
    return result


def parse_prediction(value):
    if isinstance(value, dict):
        return value
    text = str(value or "")
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_event_type(item):
    return item.get("event_type")


def get_incident_level(item):
    value = item.get("incident_level")
    if value is not None:
        return value
    value = item.get("event_level")
    if value is not None:
        return value
    return item.get("severity")


def get_description(item):
    return item.get("description") or item.get("event_description")


def get_scene_region(item):
    value = item.get("location") or item.get("scene_region")
    if isinstance(value, dict):
        return value.get("scene_region")
    return value


def get_road_location(item):
    value = item.get("location") or item.get("road_location")
    if isinstance(value, dict):
        return value.get("road_location")
    return value


def get_affected_lanes(item):
    return item.get("affected_lanes")


def get_environment(item):
    return item.get("environment") or item


def get_weather(item):
    env = get_environment(item)
    return env.get("weather") if isinstance(env, dict) else None


def get_visibility(item):
    env = get_environment(item)
    return env.get("visibility") if isinstance(env, dict) else None


def get_lighting(item):
    env = get_environment(item)
    return env.get("lighting") if isinstance(env, dict) else None


def get_queue_present(item):
    queue = item.get("queue")
    if isinstance(queue, dict):
        return queue.get("present")
    return item.get("queue_present")


def get_queue_overflow(item):
    queue = item.get("queue")
    if isinstance(queue, dict):
        return queue.get("overflow")
    return item.get("queue_overflow")


def get_queue_count(item):
    queue = item.get("queue")
    if isinstance(queue, dict):
        return queue.get("count")
    return item.get("queue_count")


def get_impact(item):
    return item.get("impact")


def field_report(pairs, normalizer, labels=None):
    result = {
        "n": len(pairs),
        "accuracy": accuracy(pairs),
        "macro_f1": None,
        "qwk": None,
        "per_class": None,
    }
    if pairs:
        macro = macro_f1(pairs)
        result["macro_f1"] = macro["macro_f1"]
        result["per_class"] = macro["per_class"]
        if labels:
            result["qwk"] = qwk(pairs, labels)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="evaluation/structured.json")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--subset", default="shandong_4qa")
    parser.add_argument("--out", default="evaluation/report")
    parser.add_argument("--description_metric", default="char_rouge_l", choices=["char_rouge_l", "char_bigram"])
    parser.add_argument("--description_threshold", type=float, default=0.5)
    args = parser.parse_args()

    gold_data = load_json(args.gold)
    gold_videos = [v for v in gold_data["videos"] if v.get("scorable")]
    if args.split != "all":
        gold_videos = [v for v in gold_videos if v.get("split") == args.split]
    if args.subset:
        gold_videos = [v for v in gold_videos if v.get("subset") == args.subset]

    pred_data = load_json(args.pred)
    if isinstance(pred_data, dict):
        pred_items = pred_data.get("predictions", [])
    else:
        pred_items = pred_data
    pred_map = {}
    parse_errors = 0
    for item in pred_items:
        video = item.get("video")
        raw = item.get("prediction", item.get("output", item.get("text")))
        parsed = parse_prediction(raw)
        if parsed is None:
            parse_errors += 1
            parsed = {}
        pred_map[video] = parsed

    event_pairs = []
    for item in gold_videos:
        pred = pred_map.get(item["video"], {})
        event_pairs.append((item["event_type"], normalize_event_type(pred.get("event_type"))))

    incident_pairs = []
    scene_pairs = []
    road_gold, road_pred = [], []
    lane_gold, lane_pred = [], []
    weather_pairs, visibility_pairs, lighting_pairs = [], [], []
    queue_pairs, overflow_pairs = [], []
    count_gold, count_pred = [], []
    impact_pairs = []
    description_scores = []
    env_joint = []
    location_pairs = []
    environment_pairs = []

    for item in gold_videos:
        pred = pred_map.get(item["video"], {})
        incident_pairs.append((item["incident_level"], normalize_level(get_incident_level(pred))))
        gold_scene = item["location"]["scene_region"]
        pred_scene = normalize_scene_region(get_scene_region(pred))
        if gold_scene != "未知":
            scene_pairs.append((gold_scene, pred_scene))

        gold_road = set(item["location"]["road_location"])
        if gold_road and gold_road != {"未知"}:
            pred_road = set(normalize_road_location(get_road_location(pred)))
            gold_road.discard("未知")
            pred_road.discard("未知")
            pred_road = pred_road or {"未知"}
            road_gold.append(gold_road)
            road_pred.append(pred_road)
            if gold_scene != "未知":
                location_pairs.append((
                    (gold_scene, tuple(sorted(gold_road))),
                    (pred_scene, tuple(sorted(pred_road))),
                ))

        gold_lanes = set(item["affected_lanes"])
        if gold_lanes and gold_lanes != {"未知"}:
            pred_lanes = set(normalize_affected_lanes(get_affected_lanes(pred)))
            pred_lanes.discard("未知")
            lane_gold.append(gold_lanes)
            lane_pred.append(pred_lanes or {"未知"})

        gold_weather = item["environment"]["weather"]
        gold_visibility = item["environment"]["visibility"]
        gold_lighting = item["environment"]["lighting"]
        pred_weather = normalize_weather(get_weather(pred))
        pred_visibility = normalize_visibility(get_visibility(pred))
        pred_lighting = normalize_lighting(get_lighting(pred))
        if gold_weather != "未知":
            weather_pairs.append((gold_weather, pred_weather))
        if gold_visibility is not None and gold_visibility != "未知":
            visibility_pairs.append((gold_visibility, pred_visibility))
        if gold_lighting != "未知":
            lighting_pairs.append((gold_lighting, pred_lighting))
        if gold_weather != "未知" and gold_visibility is not None and gold_visibility != "未知" and gold_lighting != "未知":
            env_joint.append(
                gold_weather == pred_weather
                and gold_visibility == pred_visibility
                and gold_lighting == pred_lighting
            )
            environment_pairs.append((
                (gold_weather, gold_visibility, gold_lighting),
                (pred_weather, pred_visibility, pred_lighting),
            ))
        gold_present = item["queue"]["present"]
        if gold_present is not None:
            queue_pairs.append((gold_present, normalize_bool(get_queue_present(pred))))
        gold_overflow = item["queue"]["overflow"]
        if gold_overflow is not None:
            overflow_pairs.append((gold_overflow, normalize_bool(get_queue_overflow(pred))))
        gold_count = item["queue"]["count"]
        if gold_count is not None:
            pred_count = normalize_count(get_queue_count(pred))
            if pred_count is not None:
                count_gold.append(gold_count)
                count_pred.append(pred_count)
        impact_pairs.append((item["impact"], normalize_level(get_impact(pred))))
        pred_desc = pred.get("event_description", pred.get("description", ""))
        if item.get("description"):
            if args.description_metric == "char_bigram":
                description_scores.append(char_bigram_f1(item["description"], pred_desc))
            else:
                description_scores.append(char_rouge_l(item["description"], pred_desc))

    report = {
        "meta": {
            "gold": args.gold,
            "pred": args.pred,
            "split": args.split,
            "subset": args.subset,
            "n_videos": len(gold_videos),
            "n_predictions": len(pred_map),
            "parse_errors": parse_errors,
        },
        "fields": {
            "event_type": field_report(event_pairs, normalize_event_type),
            "incident_level": field_report(incident_pairs, normalize_level, INCIDENT_LEVELS),
            "scene_region": field_report(scene_pairs, normalize_scene_region),
            "road_location": multilabel_metrics(road_gold, road_pred),
            "location": composite_report(location_pairs),
            "affected_lanes": multilabel_metrics(lane_gold, lane_pred),
            "weather": field_report(weather_pairs, normalize_weather),
            "visibility": field_report(visibility_pairs, normalize_visibility),
            "lighting": field_report(lighting_pairs, normalize_lighting),
            "environment": composite_report(environment_pairs),
            "environment_joint": {"n": len(env_joint), "accuracy": sum(env_joint) / len(env_joint) if env_joint else None},
            "queue_present": field_report(queue_pairs, normalize_bool),
            "queue_overflow": field_report(overflow_pairs, normalize_bool),
            "queue_count": {
                "n": len(count_gold),
                "mae": sum(abs(g - p) for g, p in zip(count_gold, count_pred)) / len(count_gold) if count_gold else None,
                "rmse": math.sqrt(sum((g - p) ** 2 for g, p in zip(count_gold, count_pred)) / len(count_gold)) if count_gold else None,
            },
            "impact": field_report(impact_pairs, normalize_level, IMPACTS),
            "description": {
                "n": len(description_scores),
                "metric": args.description_metric,
                "mean": sum(description_scores) / len(description_scores) if description_scores else None,
                "threshold": args.description_threshold,
                "pass_rate": sum(1 for s in description_scores if s >= args.description_threshold) / len(description_scores) if description_scores else None,
            },
        },
    }

    metrics = report["fields"]
    paper_six = {
        "incident_level": metrics["incident_level"].get("macro_f1"),
        "incident_type": metrics["event_type"].get("macro_f1"),
        "incident_description": metrics["description"].get("pass_rate"),
        "incident_location": metrics["location"].get("macro_f1"),
        "environmental_conditions": metrics["environment"].get("macro_f1"),
        "impact_assessment": metrics["impact"].get("macro_f1"),
    }
    paper_six_accuracy = {
        "incident_level": metrics["incident_level"].get("accuracy"),
        "incident_type": metrics["event_type"].get("accuracy"),
        "incident_description": metrics["description"].get("pass_rate"),
        "incident_location": metrics["location"].get("accuracy"),
        "environmental_conditions": metrics["environment_joint"].get("accuracy"),
        "impact_assessment": metrics["impact"].get("accuracy"),
    }
    report["paper_six"] = paper_six
    report["paper_six_accuracy"] = paper_six_accuracy
    numeric = [v for v in paper_six.values() if isinstance(v, (int, float))]
    report["paper_six_macro_average"] = sum(numeric) / len(numeric) if numeric else None
    f1_values = [
        paper_six[key]
        for key in ("incident_level", "incident_type", "incident_location", "environmental_conditions", "impact_assessment")
        if isinstance(paper_six[key], (int, float))
    ]
    report["paper_six_f1_macro_average"] = sum(f1_values) / len(f1_values) if f1_values else None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path = out.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Structured Evaluation Report",
        "",
        f"- split: {args.split}",
        f"- subset: {args.subset}",
        f"- videos: {len(gold_videos)}",
        f"- parse errors: {parse_errors}",
        "",
        "| Paper metric | Score |",
        "|---|---:|",
    ]
    for key, value in paper_six.items():
        shown = "N/A" if value is None else f"{value:.4f}"
        lines.append(f"| {key} | {shown} |")
    overall = report["paper_six_macro_average"]
    f1_overall = report["paper_six_f1_macro_average"]
    lines.append(f"| paper_six_macro_average | {'N/A' if overall is None else f'{overall:.4f}'} |")
    lines.append(f"| paper_six_f1_macro_average | {'N/A' if f1_overall is None else f'{f1_overall:.4f}'} |")
    lines += ["", "| Paper metric | Accuracy |", "|---|---:|"]
    for key, value in paper_six_accuracy.items():
        shown = "N/A" if value is None else f"{value:.4f}"
        lines.append(f"| {key} | {shown} |")
    lines += ["", "| Field | n | Accuracy | Macro-F1 | QWK / Micro-F1 |", "|---|---:|---:|---:|---:|"]
    for name, item in metrics.items():
        n = item.get("n")
        acc = item.get("accuracy")
        macro = item.get("macro_f1")
        extra = item.get("qwk", item.get("micro_f1"))
        lines.append(
            f"| {name} | {n} | {'N/A' if acc is None else f'{acc:.4f}'} | "
            f"{'N/A' if macro is None else f'{macro:.4f}'} | "
            f"{'N/A' if extra is None else f'{extra:.4f}'} |"
        )
    desc = metrics.get("description", {})
    if desc.get("pass_rate") is not None:
        lines.append(
            f"\nDescription pass rate: {desc['pass_rate']:.4f} "
            f"(metric={desc.get('metric')}, threshold={desc.get('threshold')}, mean={desc.get('mean'):.4f})"
        )
    report_md = out.with_suffix(".md")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(report_path)
    print(report_md)


if __name__ == "__main__":
    main()
