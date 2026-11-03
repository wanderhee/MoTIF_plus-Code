MIN_SPEED_WEIGHT=0.02
SLOW_SPEED_WEIGHT=0.25
MAX_JAM_VEHICLE_NUM=20
MIN_VEHICLE_WIDTH = 1.5
MAX_VEHICLE_GAP_WEIGHT = 3
STOP_DURATION_SECONDS = 3.0
BREAKDOWN_DURATION_SECONDS = 180.0

VEHICLE_NORMALIZATION_FACTOR = {
    'car': 1.2,
    'truck': 0.5,
    'motorcycle': 1.3,
    'default': 1.0
}

MIN_DETECTION_SIZE = 10.0
MAX_DETECTION_SIZE = 200.0


PEDESTRIAN_CLASSES = {"people"}
MOTOR_VEHICLE_CLASSES = {"car", "truck", "excavator"}
NON_MOTOR_VEHICLE_CLASSES = {"bike", "motorcycle", "tricycle", "e-bike"}
VEHICLE_CLASSES = MOTOR_VEHICLE_CLASSES | NON_MOTOR_VEHICLE_CLASSES


class Judger:
    def __init__(self,current_data,prev_data,result,jam_vehicle_info):
        self.current_data=current_data
        self.prev_data=prev_data
        if result is None:
            self.result = [False, False, False, False]
        else:
            self.result = list(result)
            while len(self.result) < 4:
                self.result.append(False)
        self.jam_vehicle_info = jam_vehicle_info
        self.jam_axis = getattr(self, 'jam_axis', 'x')
        self.roi = getattr(self, 'roi', None)
        self.min_speed = self._calculate_min_speed()
        self.slow_speed = self._calculate_slow_speed()

    def main(self):
        if self.current_data is None:
            return
        
        self.min_speed = self._calculate_min_speed()
        self.slow_speed = self._calculate_slow_speed()
        fps = float(self.current_data.get('fps', 0) or 0)
        if fps > 0:
            self.min_speed *= fps
            self.slow_speed *= fps
        if self.is_slow_vehicle():
            x = self.current_data['x']
            y = self.current_data['y']
            in_roi = True
            if self.roi is not None:
                x1, y1, x2, y2 = self.roi
                in_roi = (x >= x1 and x <= x2 and y >= y1 and y <= y2)
            if in_roi:
                vehicle_info = {
                    'x': x,
                    'y': y,
                    'width': self.current_data['size_w']
                }
                self.jam_vehicle_info.append(vehicle_info)

        if self.result[0]==False:
            if self.isParking():
                self.result[1]=True
                if self.isBreakdown():
                    self.result[3]=True
            else:
                self.result[1]=False
        
        if self.isPeople():
            self.result[2]=True
    
    def checkJam(self):
        if self.isJam():
            self.result[0] = True
            self.result[1] = False
            return True
        return False

    def isParking(self):
        valid_vehicle = self.current_data['class'] in MOTOR_VEHICLE_CLASSES
        if not valid_vehicle:
            return False

        if len(self.prev_data) == 0:
            self.current_data['low_speed_duration'] = 0.0
            return False

        speed_threshold = self.min_speed

        current_frame = self.current_data.get('frame', None)
        prev_frame = self.prev_data.get('frame', None)
        fps = self.current_data.get('fps', None)
        if current_frame is not None and prev_frame is not None and fps and fps > 0:
            dt = (current_frame - prev_frame) / float(fps)
        else:
            prev_time = self.prev_data.get('Time', self.current_data['Time'])
            dt = self.current_data['Time'] - prev_time
        if dt < 0:
            dt = 0.0

        prev_duration = float(self.prev_data.get('low_speed_duration', 0.0))
        is_low_speed = self.current_data['speed'] <= speed_threshold
        current_duration = prev_duration + dt if is_low_speed else 0.0
        self.current_data['low_speed_duration'] = current_duration

        return is_low_speed and current_duration >= STOP_DURATION_SECONDS
        
    def isJam(self):
        if len(self.jam_vehicle_info) < MAX_JAM_VEHICLE_NUM:
            return False

        axis = 'x' if self.jam_axis not in ('x', 'y') else self.jam_axis
        
        sorted_vehicles = sorted(self.jam_vehicle_info, key=lambda v: v[axis])

        consecutive_count = 1
        for i in range(1, len(sorted_vehicles)):
            prev = sorted_vehicles[i-1]
            curr = sorted_vehicles[i]

            axis_distance = abs(curr[axis] - prev[axis])
            gap_threshold = min(prev['width'], curr['width']) * MAX_VEHICLE_GAP_WEIGHT

            if axis_distance < gap_threshold:
                consecutive_count += 1
                if consecutive_count >= MAX_JAM_VEHICLE_NUM:
                    return True
            else:
                consecutive_count = 1

        return False
    
    def isPeople(self):
        return (self.current_data['class'] in PEDESTRIAN_CLASSES and
                self.current_data.get('confidence', 0.0) >= 0.8)
    
    def isBreakdown(self):
        if not self.isParking():
            return False
        
        current_duration = float(self.current_data.get('low_speed_duration', 0.0))
        
        return current_duration >= BREAKDOWN_DURATION_SECONDS
    
    def _calculate_min_speed(self):
        if self.current_data is None:
            return MIN_VEHICLE_WIDTH * MIN_SPEED_WEIGHT
        
        vehicle_class = self.current_data.get('class', 'default')
        size = self.current_data.get('size_h', MIN_VEHICLE_WIDTH) if vehicle_class == "truck" else self.current_data.get('size_w', MIN_VEHICLE_WIDTH)
        
        normalization_factor = VEHICLE_NORMALIZATION_FACTOR.get(
            vehicle_class, 
            VEHICLE_NORMALIZATION_FACTOR['default']
        )
        
        normalized_size = size * normalization_factor
        
        if normalized_size < MIN_DETECTION_SIZE:
            normalized_size = MIN_DETECTION_SIZE
        elif normalized_size > MAX_DETECTION_SIZE:
            normalized_size = MAX_DETECTION_SIZE
        
        return normalized_size * MIN_SPEED_WEIGHT
    
    def _calculate_slow_speed(self):
        if self.current_data is None:
            return MIN_VEHICLE_WIDTH * SLOW_SPEED_WEIGHT
        
        size_w = self.current_data.get('size_w', MIN_VEHICLE_WIDTH)
        vehicle_class = self.current_data.get('class', 'default')
        
        normalization_factor = VEHICLE_NORMALIZATION_FACTOR.get(
            vehicle_class, 
            VEHICLE_NORMALIZATION_FACTOR['default']
        )
        
        normalized_size = size_w * normalization_factor
        
        if normalized_size < MIN_DETECTION_SIZE:
            normalized_size = MIN_DETECTION_SIZE
        elif normalized_size > MAX_DETECTION_SIZE:
            normalized_size = MAX_DETECTION_SIZE
        
        return normalized_size * SLOW_SPEED_WEIGHT
    
    def is_slow_vehicle(self):
        return (self.current_data['class'] in VEHICLE_CLASSES and
                self.current_data['speed'] < self.slow_speed)
