import cv2  
import mediapipe as mp  
import time
import subprocess
import sys
import threading
from pathlib import Path


def osascript(script: str) -> None:
    # macOS only
    subprocess.run(
        ["osascript", "-e", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

class VideoPlayer:
    def __init__(self):
        self.process = None
        self.is_macos = sys.platform == "darwin"

    def play_video(self, video_path: Path) -> None:
        # Play video based on platform
        if self.is_macos:
            self._play_video_macos(video_path)
        else:
            self._play_video_windows(video_path)

    def close_video(self, video_path: Path) -> None:
        # Close video based on platform
        if self.is_macos:
            self._close_video_macos(video_path)
        else:
            self._close_video_windows()



    def _play_video_windows(self, video_path: Path) -> None:
        # Play video with opencv on windows
        if self.process is not None:
            return

        video_name = video_path.name

        def play_in_thread():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return

            window_name = 'Video'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 390, 780)
            cv2.moveWindow(window_name, 25, 45)

            fps =  30
            delay = int(1000 / fps)

            while cap.isOpened() and self.process is not None:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                cv2.imshow(window_name, frame)
                if cv2.waitKey(delay) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyWindow(window_name)


        self.process = threading.Thread(target=play_in_thread, daemon=True)
        self.process.start()


    def _close_video_windows(self) -> None:
        if self.process is not None:
            self.process = None
            try:
                cv2.destroyAllWindows()
            except:
                pass

    def _play_video_macos(self, video_path: Path) -> None:
        absolute_path = str(video_path)
        script = f'''
        tell application "QuickTime Player"
            activate
            set doc to open POSIX file "{absolute_path}"
            
            tell doc
                play
                set presenting to false
                tell front window
                    set bounds to {25, 45, 415, 825}
                end tell
    
            end tell
        end tell
        '''
        osascript(script)


    def _close_video_macos(self, video_path: Path) -> None:
        video_name = video_path.name
        script = f'''
        tell application "QuickTime Player"
            repeat with d in documents
                try
                    if (name of d) is "{video_name}" then
                        stop d
                        close d saving no
                    end if
                end try
            end repeat
        end tell
        '''
        osascript(script)

def draw_warning(frame, text="lock in twin"):
    h, w = frame.shape[:2]
    box_w, box_h = 500, 70
    x1 = (w - box_w) // 2
    y1 = 24
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 0, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (80, 255, 160) , 4)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 160) , 2)

    cv2.putText(
        frame,
        text.upper(),
        (x1 + 26, y1 + 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


    
def main():
    timer = 2.0
    looking_down_threshold = 0.25
    debounce_threshold = 0.45
    
    skyrim_skeleton_video = Path("./assets/skyrim-skeleton.mp4").resolve()
    if not skyrim_skeleton_video.exists():
        print("Could not open skyrim-skeleton.mp4")
        return
    
    face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Could not open webcam")
        return

    player = VideoPlayer()
    doomscroll = None
    video_playing = False

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        height, width, depth = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_image = face_mesh_landmarks.process(rgb_frame)
        face_landmark_points = processed_image.multi_face_landmarks

        current = time.time()

        if face_landmark_points:
            one_face_landmark_points = face_landmark_points[0].landmark
            
            left = [one_face_landmark_points[145], one_face_landmark_points[159]]
            for landmark_point in left:
                x = int(landmark_point.x * width)
                y = int(landmark_point.y * height)

            right = [one_face_landmark_points[374], one_face_landmark_points[386]]
            for landmark_point in right:
                    x = int(landmark_point.x * width)
                    y = int(landmark_point.y * height)
                 
            
            lx = int((left[0].x + left[1].x) / 2 * width)
            ly = int((left[0].y + left[1].y) / 2 * height)

            rx = int((right[0].x + right[1].x) / 2 * width)
            ry = int((right[0].y + right[1].y) / 2 * height)

            box = 50

            cv2.rectangle(frame, (lx - box, ly - box), (lx + box, ly + box), (10, 255, 0), 2)
            cv2.rectangle(frame, (rx - box, ry - box), (rx + box, ry + box), (10, 255, 0), 2)
            

            l_iris = one_face_landmark_points[468]
            r_iris = one_face_landmark_points[473]
            
            l_ratio = (l_iris.y  - left[1].y)  / (left[0].y  - left[1].y  + 1e-6)
            r_ratio = (r_iris.y - right[1].y) / (right[0].y - right[1].y + 1e-6)

            avg_ratio = (l_ratio + r_ratio) / 2.0

            if video_playing:
                is_looking_down = avg_ratio < debounce_threshold
            else:
                is_looking_down = avg_ratio < looking_down_threshold


            if is_looking_down:
                if doomscroll is None:
                    doomscroll = current

                if (current - doomscroll) >= timer:               
                    if not video_playing:
                        player.play_video(skyrim_skeleton_video)
                        video_playing = True

            else:
                doomscroll = None
                if video_playing:
                    player.close_video(skyrim_skeleton_video)
                    video_playing = False
        else:
            doomscroll = None
            if video_playing:
                player.close_video(skyrim_skeleton_video)
                video_playing = False

        if video_playing:
            draw_warning(frame, "doomscrolling alarm")

        cv2.imshow('lock in', frame)
        key = cv2.waitKey(1)

        if key == 27:
            break

    if video_playing:
        player.close_video(skyrim_skeleton_video)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



