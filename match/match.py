import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from scipy.spatial import procrustes


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


class PosePipline:
    def __init__(self, target_path = './match/weien.mp4', max_step = 10):
        self.target = read_video(target_path)
        self.w2, self.h2 = self.target[0].shape[1], self.target[0].shape[0]
        self.max_step = max_step
        self.init()
        # 预处理
        model = self.mp_pose.Pose()
        self.target_pose = []
        for frame in tqdm(self.target):
            pose = self.detect_pose(model, frame)
            self.target_pose.append(pose)
        for i, frame in enumerate(self.target):
            self.draw(frame, self.target_pose[min(i+10, len(self.target_pose)-1)])

    # 重启
    def init(self): 
        self.now_frame = 0
        self.batch = []
        self.score = 0
        self.mp_pose = mp.solutions.pose
        self.pose_landmark = self.mp_pose.PoseLandmark
        self.mp_drawing = mp.solutions.drawing_utils
        self.model = self.mp_pose.Pose()

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
            body1.append([action[i].x * self.w1 , action[i].y * self.h1])
            body2.append([target[i].x * self.w2, target[i].y * self.h2])
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
    def __call__(self, frame):
        self.w1, self.h1 = frame.shape[1], frame.shape[0]
        pose = self.detect_pose(self.model, frame)
        self.batch.append(pose)
        if len(self.batch) == self.max_step:
            self.score = self.match_batch(self.batch, self.target_pose[self.now_frame-self.max_step+1:self.now_frame+1])
            self.batch = []
        self.draw(frame, pose)
        self.now_frame = min(len(self.target)-1, self.now_frame+1)
        return frame, self.target[self.now_frame], self.score


##########################


# main
def compare():
    w, h = 270, 480
    model = PosePipline('./match/weizhi.mp4')
    action = read_video('./match/weien.mp4')[30:]
    for frame in action:        
        frame1, frame2, score = model(frame)
        frame1, frame2 = cv2.resize(frame1, (w, h)), cv2.resize(frame2, (w, h))
        frame = cv2.hconcat([frame1, frame2])
        cv2.putText(frame, str(round(score,1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == '__main__':
    # compare()
    cap = cv2.VideoCapture(0)
    w1, w2, h = 640, 270, 480
    output_file = "./match/pose_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, (w1+w2, h))
    model = PosePipline('./match/weien.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)    
        frame1, frame2, score = model(frame)
        frame1, frame2 = cv2.resize(w1, h), cv2.resize(frame2, (w2, h))
        frame = cv2.hconcat([frame1, frame2])
        cv2.putText(frame, str(round(score,1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        out.write(frame)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    out.release()
    cv2.destroyAllWindows()