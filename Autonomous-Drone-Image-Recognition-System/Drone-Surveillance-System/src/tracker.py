import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
from .config import LOCK_SIZE_THRESHOLD, ANOMALY_BUFFER_SIZE, LOCK_DURATION_FRAMES

class ObjectAnalyzer:
    def __init__(self):
        # Stores history of object features (Size)
        self.feature_history = deque(maxlen=2000)
        # ML Model to detect anomalies
        self.clf = IsolationForest(contamination=0.05, random_state=42)
        self.is_fitted = False
        
        # Track active locks: {track_id: frames_remaining}
        self.lock_timers = {} 

    def extract_features(self, box, frame_shape):
        """Extracts normalized area from bounding box."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        img_area = frame_shape[0] * frame_shape[1]
        
        norm_area = (w * h) / img_area
        return [norm_area]

    def update_locks(self, tracks, frame_shape):
        """
        1. Extract data from objects.
        2. Train ML model (if enough data).
        3. Decide which objects to LOCK.
        """
        metadata = {}
        current_features = []
        
        # 1. Feature Extraction
        for track in tracks:
            box = track.xyxy[0].cpu().numpy()
            feats = self.extract_features(box, frame_shape)
            current_features.append(feats)
            self.feature_history.append(feats)

        # 2. Train Anomaly Detector (Online Learning)
        if len(self.feature_history) >= ANOMALY_BUFFER_SIZE and not self.is_fitted:
            self.clf.fit(list(self.feature_history))
            self.is_fitted = True

        # 3. Apply Locking Rules
        for i, track in enumerate(tracks):
            track_id = int(track.id)
            norm_area = current_features[i][0]
            
            status = "SCANNING"
            is_locked = False

            # Check existing lock
            if track_id in self.lock_timers:
                self.lock_timers[track_id] -= 1
                if self.lock_timers[track_id] > 0:
                    status = f"LOCKED ({self.lock_timers[track_id]})"
                    is_locked = True
                else:
                    del self.lock_timers[track_id] # Expired
            
            # Check for new lock triggers
            if not is_locked:
                trigger = False
                
                # Trigger A: Large Object
                if norm_area > LOCK_SIZE_THRESHOLD:
                    status = "TARGET (SIZE)"
                    trigger = True
                
                # Trigger B: Anomaly (ML)
                elif self.is_fitted:
                    if self.clf.predict([current_features[i]])[0] == -1:
                        status = "ANOMALY"
                        trigger = True
                
                if trigger:
                    self.lock_timers[track_id] = LOCK_DURATION_FRAMES
                    is_locked = True

            metadata[track_id] = {"locked": is_locked, "status": status}
            
        return metadata
