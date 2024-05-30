from tracker import Tracker
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

class DeepSortTracker(Tracker):
    def __init__(self, nms:float) -> None:
        super().__init__()
        self.object_tracker = DeepSort(nms_max_overlap=nms)

    def track(self, boxes:np.array, roi:np.array, frame):
        tracker_results = [
            ([box.xywh[0][0], 
            box.xywh[0][1], 
            box.xywh[0][2], 
            box.xywh[0][3]], 
            box.conf[0], 
            box.cls[0]) 
            for box in boxes]

        inside_results = []
        for result in tracker_results:
            center_x = result[0][0] + result[0][2] * 0.5
            center_y = result[0][1] + result[0][3] * 1.0

            if cv2.pointPolygonTest(roi, (center_x, center_y), False) <= 0:
                inside_results.append(result)
            
        tracks = self.object_tracker.update_tracks(inside_results, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            cv2.circle(frame, (int(center_x), int(center_y)), 1, (0, 255, 0), 2)
        
        return frame