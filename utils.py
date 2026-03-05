import cv2
import numpy as np

def draw_status(frame, status, name=None):
    """
    Draws status icons and text on the frame.
    status: 'SUCCESS', 'ALREADY_ATTENDED', 'UNRECOGNIZED', None
    """
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    if status == 'SUCCESS':
        color = (0, 255, 0)
        text = f"Berhasil Absen: {name}"
        icon_pos = (w // 2, h // 2)
        cv2.circle(overlay, icon_pos, 50, color, -1)
        cv2.putText(overlay, "V", (icon_pos[0]-20, icon_pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
    elif status == 'ALREADY_ATTENDED':
        color = (0, 255, 255)
        text = f"Sudah Absen: {name}"
        icon_pos = (w // 2, h // 2)
        cv2.circle(overlay, icon_pos, 50, color, -1)
        cv2.putText(overlay, "V", (icon_pos[0]-20, icon_pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    elif status == 'UNRECOGNIZED':
        color = (0, 0, 255)
        text = "Wajah Tidak Dikenali"
        icon_pos = (w // 2, h // 2)
        cv2.circle(overlay, icon_pos, 50, color, -1)
        cv2.putText(overlay, "X", (icon_pos[0]-20, icon_pos[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    else:
        return frame

    cv2.putText(overlay, text, (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    return frame
