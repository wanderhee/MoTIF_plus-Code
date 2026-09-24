import json
import logging
import os

logger = logging.getLogger(__name__)


class PromptManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.event_type_map = {}
        self.event_level_map = {}
        self.special_event_prompts = {}
        self.base_prompt_template = ""
        self.unknown_event_prompt_template = ""
        self.load_config()

    def load_config(self):
        logger.info(f"开始加载提示词配置文件: {self.config_path}")
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"配置文件不存在: {self.config_path}")
                return False

            with open(self.config_path, 'r', encoding='utf-8') as f:
                prompts_config = json.load(f)

            self.event_type_map = {int(k): v for k, v in prompts_config.get("event_type_map", {}).items()}
            self.event_level_map = prompts_config.get("event_level_map", {})
            self.special_event_prompts = prompts_config.get("special_event_prompts", {})
            self.base_prompt_template = prompts_config.get("base_prompt_template", "")
            self.unknown_event_prompt_template = prompts_config.get("unknown_event_prompt_template", "")

            logger.info(f"提示词配置加载成功")
            logger.info(f"   事件类型数量: {len(self.event_type_map)}")
            logger.info(f"   事件级别数量: {len(self.event_level_map)}")
            logger.info(f"   特殊事件提示词数量: {len(self.special_event_prompts)}")
            logger.info(f"   base_prompt_template长度: {len(self.base_prompt_template)}字符")

            for i, (event_id, event_name) in enumerate(list(self.event_type_map.items())[:5]):
                logger.debug(f"   事件ID {event_id}: {event_name}")

            return True

        except Exception as e:
            logger.error(f"加载提示词配置失败: {str(e)}")
            import traceback
            logger.error(f"堆栈跟踪: {traceback.format_exc()}")
            return False

    def get_structured_prompt(self, event_type, media_type):
        logger.debug(f"获取结构化提示词 - 事件类型: {event_type}, 媒体类型: {media_type}")

        event_type_name = self.event_type_map.get(event_type, f"未知事件类型({event_type})")
        logger.debug(f"   事件类型名称: {event_type_name}")

        if event_type_name in self.special_event_prompts:
            event_specific_prompt = self.special_event_prompts[event_type_name]
            logger.debug(f"   使用特殊事件提示词")
        elif event_type_name:
            try:
                event_specific_prompt = self.base_prompt_template.format(
                    media_type=media_type,
                    event_type=event_type_name
                )
                logger.debug(f"   使用基础提示词模板，media_type已替换")
            except KeyError as e:
                logger.error(f"模板格式化KeyError: {e}")
                event_specific_prompt = self.base_prompt_template
                if "{media_type}" in event_specific_prompt:
                    event_specific_prompt = event_specific_prompt.replace("{media_type}", media_type)
                if "{event_type}" in event_specific_prompt:
                    event_specific_prompt = event_specific_prompt.replace("{event_type}", event_type_name)
        else:
            try:
                event_specific_prompt = self.unknown_event_prompt_template.format(
                    media_type=media_type,
                    event_type=event_type
                )
            except:
                event_specific_prompt = self.unknown_event_prompt_template
                if "{media_type}" in event_specific_prompt:
                    event_specific_prompt = event_specific_prompt.replace("{media_type}", media_type)
                if "{event_type}" in event_specific_prompt:
                    event_specific_prompt = event_specific_prompt.replace("{event_type}", str(event_type))

        if "{media_type}" in event_specific_prompt:
            logger.warning(f"提示词中仍有未替换的{{media_type}}，进行手动替换")
            event_specific_prompt = event_specific_prompt.replace("{media_type}", media_type)

        structured_prompt = f"""{event_specific_prompt}

【输出格式要求】
请严格按以下JSON格式输出分析结果，所有字段值都必须是字符串类型：
{{
    "has_accident": true/false,
    "accident_type": "填写事件类型：异常停车/非机动车闯入/行人闯入/无事件",
    "highway_info": "高速公路信息，格式必须为：高速代码+名称+桩号，如：G22青兰高速K081+154",
    "event_description": "详细的事件描述（如果无事件，请描述正常交通状况）",
    "vehicles_info": "车辆信息描述",
    "personnel_info": "人员信息描述",
    "position_info": "位置信息描述",
    "infected_lane": "影响车道描述",
    "environment_condition": "环境状况描述，格式必须为：天气:[描述]，能见度:[描述]，光照:[描述]",
    "queue_situation": "排队情况描述",
    "summary": "事件总结和分析建议",
    "event_time": "从视频/图片中识别的事件发生时间，格式：YYYY-MM-%d HH:MM:SS"
}}

【字段内容要素要求】
1. highway_info:
   - 必须包含：高速代码+名称+桩号
   - 示例：G22青兰高速K081+154
   - 禁止：不要添加省份前缀

2. vehicles_info:
   - 有事件时必须包含：车辆类型、颜色、特征、车牌信息（如果可见）
   - 无事件时使用："无事件发生，车辆正常通行"

3. personnel_info:
   - 有事件时必须包含：人员数量、大致身份、关键动作和行为
   - 无事件时使用："无人员异常活动"

4. position_info:
   - 有事件时必须包含：具体车道位置（应急车道/行车道/超车道等）、在画面中的相对位置（左侧/右侧/中央等）
   - 无事件时使用："各车道车辆正常通行"

5. environment_condition:
   - 必须包含：天气状况、能见度情况、光照条件
   - 格式必须为："天气:[描述]，能见度:[描述]，光照:[描述]"

6. infected_lane:
   - 必须描述：影响的具体车道、影响程度、对其他车道的潜在风险

7. queue_situation:
   - 必须描述：排队长度、排队原因、影响范围

【事件类型定义】
- 异常停车：车辆在非指定区域停止，包括：故障停车、违法停车、紧急停车
- 交通事故：车辆之间的碰撞事故，包括：追尾、刮擦、侧翻、多车相撞
- 非机动车闯入：自行车、电动自行车、三轮车等非机动车出现在高速公路
- 行人闯入：行人在行车道、路肩或中央分隔带活动
- 拥堵：交通流量大导致车辆排队缓行或停滞，平均车速低于20km/h
- 施工：道路施工养护作业，包括：占道施工、养护车辆作业、施工人员活动
- 无事件：未发现任何异常事件，交通流正常


【关键要求】
1. 严格遵循所有字段的格式要求，特别是highway_info和event_time
2. 请只关注高速公路范围内事件情况，监控摄像头可能会拍摄到高速公路外的建筑、农田、其他道路等区域，这些区域的情况不属于本分析范围
3. event_time必须从视频/图片中的时间信息（如时间戳、时钟等）中识别
4. 每个字段必须包含指定的内容要素
5. 只输出JSON格式，不要任何额外文本
6. 如果确认没有事件发生，has_accident必须为false，accident_type必须为"无事件"
7. 基于YOLO初步检测结果（可能{event_type_name}事件）进行分析，但最终判断要基于完整的媒体内容

请基于实际视觉内容进行详细分析，提供准确的事件检测结果。"""

        logger.debug(f"   生成的prompt长度: {len(structured_prompt)}字符")
        logger.debug(f"   Prompt前100字符: {structured_prompt[:100]}...")

        return structured_prompt

    def get_event_type_name(self, event_type):
        event_name = self.event_type_map.get(event_type, f"未知事件类型({event_type})")
        logger.debug(f"获取事件类型名称: {event_type} -> {event_name}")
        return event_name
