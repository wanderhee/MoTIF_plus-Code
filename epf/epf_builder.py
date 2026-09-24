import json


class EPFBuilder:

    DETECTION_KEYS = (
        "detections", "detectionResults", "detection_results", "objects",
        "targets", "yoloResults", "yolo_results", "boxes", "results"
    )
    CLASS_KEYS = ("class", "className", "class_name", "label", "category", "name", "cls")
    SCORE_KEYS = ("score", "confidence", "conf", "prob", "probability")
    BBOX_KEYS = ("bbox", "box", "xyxy", "rect", "boundingBox", "bounding_box")

    def __init__(self, prompt_manager):
        self.prompt_manager = prompt_manager

    @staticmethod
    def _first(mapping, keys, default=None):
        if not isinstance(mapping, dict):
            return default
        for key in keys:
            value = mapping.get(key)
            if value is not None and value != "":
                return value
        return default

    @staticmethod
    def _to_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_text(value, default="未知"):
        if value is None or value == "":
            return default
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                return str(value)
        return str(value)

    def _extract_raw_detections(self, event_data):
        if not isinstance(event_data, dict):
            return []
        for key in self.DETECTION_KEYS:
            value = event_data.get(key)
            if isinstance(value, list):
                if len(value) == 1 and isinstance(value[0], dict):
                    nested = self._extract_raw_detections(value[0])
                    if nested:
                        return nested
                return value
            if isinstance(value, dict):
                nested = self._extract_raw_detections(value)
                if nested:
                    return nested
        if any(k in event_data for k in self.BBOX_KEYS) and any(k in event_data for k in self.CLASS_KEYS):
            return [event_data]
        return []

    def _parse_bbox(self, raw_bbox, media_meta=None):
        if raw_bbox is None:
            return None, None
        x1 = y1 = x2 = y2 = None
        if isinstance(raw_bbox, dict):
            if all(k in raw_bbox for k in ("x1", "y1", "x2", "y2")):
                x1, y1, x2, y2 = [self._to_float(raw_bbox[k]) for k in ("x1", "y1", "x2", "y2")]
            elif all(k in raw_bbox for k in ("left", "top", "right", "bottom")):
                x1, y1, x2, y2 = [self._to_float(raw_bbox[k]) for k in ("left", "top", "right", "bottom")]
            elif all(k in raw_bbox for k in ("x", "y", "w", "h")):
                x1 = self._to_float(raw_bbox["x"])
                y1 = self._to_float(raw_bbox["y"])
                x2 = x1 + self._to_float(raw_bbox["w"])
                y2 = y1 + self._to_float(raw_bbox["h"])
        elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            x1, y1, x2, y2 = [self._to_float(v) for v in raw_bbox[:4]]

        if None in (x1, y1, x2, y2):
            return self._safe_text(raw_bbox), None

        raw = [x1, y1, x2, y2]
        width = self._to_float((media_meta or {}).get("width"), 0.0)
        height = self._to_float((media_meta or {}).get("height"), 0.0)
        if all(0.0 <= v <= 1.0 for v in raw):
            norm = raw
        elif width > 0 and height > 0:
            norm = [x1 / width, y1 / height, x2 / width, y2 / height]
        else:
            norm = None
        if norm is not None:
            norm = [max(0.0, min(1.0, v)) for v in norm]
        return raw, norm

    def build_detection_set(self, event_data, media_meta=None):
        detections = []
        for idx, det in enumerate(self._extract_raw_detections(event_data), 1):
            if not isinstance(det, dict):
                continue
            cls = self._first(det, self.CLASS_KEYS, default=f"object_{idx}")
            score = self._to_float(self._first(det, self.SCORE_KEYS, default=0.0), 0.0)
            bbox_raw, bbox_norm = self._parse_bbox(self._first(det, self.BBOX_KEYS, default=None), media_meta)
            attributes = det.get("attributes") or det.get("attrs") or {}
            if not isinstance(attributes, dict):
                attributes = {"description": attributes}
            for key in (
                "motion", "motion_state", "state", "lane", "lane_position",
                "direction", "movement_direction", "speed", "stop_duration",
                "track_id", "id"
            ):
                if key in det and key not in attributes:
                    attributes[key] = det[key]
            detections.append({
                "id": idx,
                "class": self._safe_text(cls),
                "bbox": bbox_raw,
                "bbox_norm": bbox_norm,
                "score": score,
                "attributes": attributes,
            })
        return detections

    def build_incident_hypothesis(self, event_data, detections):
        event_type = event_data.get("eventType", 0) if isinstance(event_data, dict) else 0
        event_type_name = self.prompt_manager.get_event_type_name(event_type)
        level_raw = self._first(event_data, ("grade", "eventLevel", "event_level", "level"), default="2")
        level_name = self.prompt_manager.event_level_map.get(str(level_raw), self._safe_text(level_raw, "未知"))

        location_parts = []
        for key in ("road", "roadName", "route", "pileNum", "stake_number", "location"):
            value = event_data.get(key) if isinstance(event_data, dict) else None
            if value not in (None, ""):
                value = str(value).strip()
                if value and value not in location_parts:
                    location_parts.append(value)

        return {
            "event_type_id": event_type,
            "event_type": event_type_name,
            "event_level_id": level_raw,
            "event_level": level_name,
            "location": " ".join(location_parts) if location_parts else "未提供",
            "objects": [d["class"] for d in detections],
            "attributes": {
                str(d["id"]): d.get("attributes", {})
                for d in detections if d.get("attributes")
            },
        }

    def build_evidence_tokens(self, detections, hypothesis):
        tokens = []
        for det in detections:
            bbox_norm = det.get("bbox_norm")
            bbox_text = "NA" if bbox_norm is None else ",".join(f"{v:.6f}" for v in bbox_norm)
            attrs = self._safe_text(det.get("attributes") or {}, default="{}")
            tokens.append(
                f'<EPF_EVIDENCE id="{det["id"]}" class="{det["class"]}" '
                f'bbox_norm="[{bbox_text}]" score="{det["score"]:.6f}" '
                f'event_type="{hypothesis["event_type"]}" '
                f'event_level="{hypothesis["event_level"]}" attrs=\'{attrs}\' />'
            )
        return tokens

    def build_scene_prompt(self, detections, hypothesis):
        lines = [
            "【P_scene｜场景证据】",
            f"YOLO初步事件假设：{hypothesis['event_type']}（eventType={hypothesis['event_type_id']}）",
            f"YOLO初步事件等级：{hypothesis['event_level']}（grade={hypothesis['event_level_id']}）",
            f"事件位置先验：{hypothesis['location']}",
            f"检测目标数量 N_det：{len(detections)}",
        ]
        if detections:
            for det in detections:
                bbox = det.get("bbox_norm")
                bbox_text = "未提供/无法归一化" if bbox is None else "[" + ", ".join(f"{v:.4f}" for v in bbox) + "]"
                attrs = self._safe_text(det.get("attributes") or {}, default="{}")
                lines.append(
                    f"- D{det['id']}: class={det['class']}; bbox_norm={bbox_text}; "
                    f"score={det['score']:.4f}; attributes={attrs}"
                )
        else:
            lines.append("- 上游YOLO请求未携带对象级bbox/score，因此D中暂无可序列化对象；不得虚构检测框。")
        return "\n".join(lines)

    @staticmethod
    def build_task_prompt(media_type):
        return f"""【P_task｜二次验证任务】
1. 以原始{media_type}视觉内容为最终判断依据，对YOLO初步事件假设进行二次验证；初步假设不是固定结论。
2. 对照检测框、类别、置信度及运动/车道属性，判断证据与视觉内容是否一致；存在冲突时应修正或否定初步假设。
3. 只分析高速公路范围内的目标和事件，排除道路外建筑、农田、支路等无关区域。
4. 完成事件类型、参与对象、位置、影响车道、环境、排队/交通影响和风险的结构化场景理解。"""

    @staticmethod
    def build_domain_prompt():
        return """【P_domain｜交通领域约束】
- 高速公路交通事件判断必须同时考虑目标类别、空间位置、时序/运动状态和道路区域语义。
- 置信度仅表示检测器对目标识别的可信程度，不等同于事件成立概率。
- 检测框是可核验空间证据；必须与原始媒体中的目标位置相互印证。
- 对行人、非机动车、停车、占用应急车道、施工、故障、拥堵等事件，应区分道路内与道路外目标。
- 当检测证据不足、遮挡严重或视觉内容与YOLO假设冲突时，应明确按视觉事实修正判断，不得机械接受YOLO结论。"""

    def build_incident_prompt(self, event_type, media_type):
        event_type_name = self.prompt_manager.get_event_type_name(event_type)
        if event_type_name in self.prompt_manager.special_event_prompts:
            text = self.prompt_manager.special_event_prompts[event_type_name]
        elif self.prompt_manager.base_prompt_template:
            text = self.prompt_manager.base_prompt_template
        else:
            text = "请结合当前事件类型先验完成事件核验与详细分析。"
        return (
            "【P_incident｜事件专项指令】\n" +
            text.replace("{media_type}", media_type).replace("{event_type}", event_type_name)
        )

    @staticmethod
    def build_output_prompt():
        return '''【P_output｜结构化输出约束】
只输出一个合法JSON对象，不要输出Markdown、解释文字或代码块。字段如下：
{
  "has_accident": true/false,
  "accident_type": "事件类型；无事件时必须为无事件",
  "highway_info": "高速代码+名称+桩号，例如G22青兰高速K081+154；不要添加省份前缀",
  "event_description": "事件或正常交通状况的详细描述",
  "vehicles_info": "车辆类型、颜色、特征、车牌（可见时）等；无事件时说明车辆正常通行",
  "personnel_info": "人员数量、身份、动作；无异常时写无人员异常活动",
  "position_info": "车道位置及画面相对位置",
  "infected_lane": "受影响具体车道、影响程度及潜在风险",
  "environment_condition": "天气:[描述]，能见度:[描述]，光照:[描述]",
  "queue_situation": "排队长度、原因和影响范围",
  "summary": "事件总结与分析建议",
  "event_time": "YYYY-MM-DD HH:MM:SS"
}
若视觉证据否定YOLO初步假设，has_accident必须按最终视觉判断输出，accident_type同步修正。'''

    def build(self, event_data, media_type, media_meta=None):
        detections = self.build_detection_set(event_data, media_meta)
        hypothesis = self.build_incident_hypothesis(event_data, detections)
        evidence_tokens = self.build_evidence_tokens(detections, hypothesis)

        evidence_block = "【Z_EPF｜区域对齐证据Token】\n" + (
            "\n".join(evidence_tokens) if evidence_tokens else
            '<EPF_EVIDENCE empty="true" reason="object-level detections not supplied" />'
        )

        final_prompt = "\n\n".join([
            "【EPF Evidence Prompt Fusion】",
            self.build_scene_prompt(detections, hypothesis),
            evidence_block,
            self.build_task_prompt(media_type),
            self.build_domain_prompt(),
            self.build_incident_prompt(hypothesis["event_type_id"], media_type),
            self.build_output_prompt(),
            "【EPF最终约束】YOLO的事件类型、等级和检测框仅作为可核验证据；最终结论必须由媒体视觉事实与证据一致性共同决定。",
        ])
        return {
            "D": detections,
            "E": hypothesis,
            "Z_EPF": evidence_tokens,
            "P_final": final_prompt,
        }
