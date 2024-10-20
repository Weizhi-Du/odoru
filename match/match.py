import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import procrustes


class PoseMatcher:
    def __init__(self, w = 270, h = 480, max_step = 10):
        self.w, self.h = w, h
        self.max_step = max_step
        self.batch_action, self.batch_target = [], []
        self.score = 0
        # 模型配置
        self.mp_pose = mp.solutions.pose
        self.pose_landmark = self.mp_pose.PoseLandmark
        self.mp_drawing = mp.solutions.drawing_utils
        self.model1, self.model2 =  self.mp_pose.Pose(), self.mp_pose.Pose()

    # 动作识别
    def detect_pose(self, model, frame):
        res = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return res

    # 单帧匹配
    def match(self, action, target):
        if action.pose_landmarks is None or target.pose_landmarks is None: return 100
        action = action.pose_landmarks.landmark
        target = target.pose_landmarks.landmark
        body1, body2 = [], []
        for i in [0, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28]:
            if action[i].visibility < 0.5 or target[i].visibility < 0.5: continue
            body1.append([action[i].x * self.w , action[i].y * self.h])
            body2.append([target[i].x * self.w, target[i].y * self.h])
        _, _, disparity = procrustes(np.array(body1), np.array(body2))
        return disparity * 1e2

    # 批量匹配
    def match_batch(self, action, target):
        best = 1e9
        for i in action:
            for j in target:
                best = min(best, self.match(i, j))
        return best

    # 绘制骨骼
    def draw(self, frame, result):
        if result.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
    # 调用
    def __call__(self, frame1, frame2):
        res1, res2 = self.detect_pose(self.model1, frame1), self.detect_pose(self.model2, frame2)
        self.batch_action.append(res1); self.batch_target.append(res2)
        if len(self.batch_target) == self.max_step:
            self.score = self.match_batch(self.batch_action, self.batch_target)
            self.batch_action, self.batch_target = [], []
        self.draw(frame1, res1); self.draw(frame2, res2)
        return frame1, frame2, self.score


##########################


# 读取视频
def read_video(path):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        print(f"无法打开视频文件: {path}")
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames


# main
def compare():
    w, h = 270, 480
    model = PoseMatcher(w, h)
    action = read_video('./match/weizhi.mp4')
    target = read_video('./match/weien.mp4')[30:]
    for i in range(max(len(action), len(target))):
        frame1, frame2 = cv2.resize(action[-1], (w, h)), cv2.resize(target[-1], (w, h))
        if i < len(action): frame1 = cv2.resize(action[i], (w, h))
        if i < len(target): frame2 = cv2.resize(target[i], (w, h))
        
        frame1, frame2, score = model(frame1, frame2)
        frame = cv2.hconcat([frame1, frame2])
        cv2.putText(frame, str(round(score,1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == '__main__':
    # compare()
    target = read_video('./match/weien.mp4')
    cap = cv2.VideoCapture(0)
    w, h = 270, 480

    output_file = "/match/pose_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, (w*2, h))

    i = 0
    model = PoseMatcher(w, h)
    while cap.isOpened():
        ret, frame1 = cap.read()
        if not ret: break
        frame1, frame2 = cv2.flip(frame1[-h:, -w:], 1), cv2.resize(target[-1], (w, h))
        if i < len(target): frame2 = cv2.resize(target[i], (w, h))
        
        frame1, frame2, score = model(frame1, frame2)
        frame = cv2.hconcat([frame1, frame2])
        cv2.putText(frame, str(round(score,1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        out.write(frame)

        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        i += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()