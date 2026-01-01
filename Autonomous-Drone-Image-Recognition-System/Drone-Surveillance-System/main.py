import cv2
import argparse
import os
import sys
from ultralytics import YOLO

# Add 'src' to system path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.tracker import ObjectAnalyzer
from src import config

def run_system(source_video, output_name):
    # 1. Path Setup
    base_path = os.path.dirname(__file__)
    video_input_path = os.path.join(base_path, 'data', 'input', source_video)
    video_output_path = os.path.join(base_path, 'data', 'output', 'videos', output_name)
    
    # 2. Validation
    if not os.path.exists(video_input_path):
        print(f"❌ Error: Video not found at {video_input_path}")
        return

    print(f"🚀 Initializing System with {config.MODEL_NAME}...")
    print(f"▶ Resolution: {config.INFERENCE_SIZE}px (High-Res)")
    
    # 3. Load Model
    try:
        model = YOLO(config.MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Video Setup
    cap = cv2.VideoCapture(video_input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    temp_out = os.path.join(base_path, 'data', 'output', 'videos', 'temp_proc.avi')
    out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    
    analyzer = ObjectAnalyzer()
    
    print("▶ Processing video...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # 5. INFERENCE
        results = model.track(frame, persist=True, 
                              imgsz=config.INFERENCE_SIZE, 
                              conf=config.CONF_THRESHOLD, 
                              augment=False, 
                              verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            
            # 6. LOGIC UPDATE
            meta_data = analyzer.update_locks(boxes, frame.shape)
            
            # 7. DRAWING
            annotated_frame = frame.copy()
            for box, track_id, cls, conf in zip(boxes.xyxy.cpu().numpy(), track_ids, boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
                x1, y1, x2, y2 = map(int, box)
                info = meta_data.get(track_id, {})
                
                is_locked = info.get('locked', False)
                status_text = info.get('status', '')
                obj_name = model.names[int(cls)]
                
                # --- COLOR LOGIC (UPDATED) ---
                if is_locked:
                    # LOCKED: Red Background, White Text
                    color = (0, 0, 255)      # Red in BGR
                    text_color = (255, 255, 255) # White
                    thickness = 3
                else:
                    # NORMAL: Bright Yellow Background, Black Text (High Visibility)
                    color = (0, 255, 255)    # Yellow in BGR
                    text_color = (0, 0, 0)   # Black
                    thickness = 2
                
                # Draw Bounding Box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw Label Background
                label = f"#{track_id} {obj_name} {conf:.2f} | {status_text}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                
                # Filled box for text
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                
                # Draw Text (using dynamic text_color)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        else:
            annotated_frame = frame

        out.write(annotated_frame)

    cap.release()
    out.release()
    
    # 8. Convert to MP4
    if os.path.exists(video_output_path): os.remove(video_output_path)
    os.system(f"ffmpeg -y -i {temp_out} -vcodec libx264 {video_output_path} -loglevel quiet")
    if os.path.exists(temp_out): os.remove(temp_out)
    
    print(f"✅ Success! Video saved at: {video_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Filename of video inside data/input/")
    parser.add_argument("--out", type=str, default="final_result.mp4", help="Output filename")
    args = parser.parse_args()
    
    run_system(args.source, args.out)
