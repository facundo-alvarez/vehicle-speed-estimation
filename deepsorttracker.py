from tracker import Tracker
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

class DeepSortTracker(Tracker):
    def __init__(self, nms:float) -> None:
        super().__init__()
        self.object_tracker = DeepSort(nms_max_overlap=nms)

    def __get_ltwh(self, track):
        x1, y1, x2, y2 = track.to_tlbr()
        w = x2 - x1
        h = y2 - y1
        return x1, y1, w, h

    def track(self, boxes:list, roi:np.array, frame):

        inside_results = []
        for result in boxes:
            if cv2.pointPolygonTest(roi, (result[0][0] + result[0][2] * 0.5, result[0][1] + result[0][3]), False) >= 0:
                inside_results.append(result)
            
        tracks = self.object_tracker.update_tracks(inside_results, frame=frame)
        tracks = [confirmed_track for confirmed_track in tracks if confirmed_track.is_confirmed() and confirmed_track.time_since_update < 1]

        ltwh_list = [self.__get_ltwh(track) for track in tracks]
        track_ids = [track.track_id for track in tracks]
        det_classes = [track.det_class for track in tracks]
        det_confs = [track.det_conf for track in tracks]

        return ltwh_list, track_ids, det_classes, det_confs