import cv2
import mediapipe as mp
from tqdm import tqdm
import numpy as np
from scipy.spatial import procrustes


# 模型配置
mp_pose = mp.solutions.pose
pose_landmark = mp_pose.PoseLandmark
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


# 读取视频
def read_video(path):
    video = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    video.release()
    return frames


# 动作识别
def detect_pose(frames):
    result = []
    for frame in tqdm(frames):
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result.append(res)
    return result


# 动作匹配 （一帧）
def match(action, target, w, h):
    action = action.pose_landmarks.landmark
    target = target.pose_landmarks.landmark
    def dist(x1, y1, x2, y2):
        return ((x1-x2)**2+(y1-y2)**2)**0.5
    def origin(landmark):
        x = (landmark[pose_landmark.LEFT_HIP].x * w + landmark[pose_landmark.RIGHT_HIP].x * w)/2
        y = (landmark[pose_landmark.LEFT_HIP].y * h + landmark[pose_landmark.RIGHT_HIP].y * h)/2
        return x, y
    # action 原点
    o1x, o1y = origin(action)
    print(o1x, o1y)
    o2x, o2y = origin(target)
    print(o2x, o2y)
    cnt, difference = 0, 0
    body1, body2 = [], []
    for i in [0, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26]:
        if action[i].visibility < 0.5 or target[i].visibility < 0.5: continue
        cnt += 1
        # difference += dist(action[i].x * w - o1x, action[i].y * h - o1y, 
        #                     target[i].x * w - o2x, target[i].y * h - o2y)
        body1.append([action[i].x * w - o1x, action[i].y * h - o1y])
        body2.append([target[i].x * w - o2x, target[i].y * h - o2y])
    mtx1, mtx2, disparity = procrustes(np.array(body1), np.array(body2))
    return disparity
        

# main
if __name__ == '__main__':
    action = read_video('weizhi.mp4')
    target = read_video('weien.mp4')[30:]
    w = int(target[0].shape[1])
    h = int(target[0].shape[0])
    for i in range(max(len(action), len(target))):
        frame1, frame2 = cv2.resize(action[-1], (w//2, h//2)), cv2.resize(target[-1], (w//2, h//2))
        if i < len(action):
            frame1 = cv2.resize(action[i], (w//2, h//2))
        if i < len(target):
            frame2 = cv2.resize(target[i], (w//2, h//2))
        frame = cv2.hconcat([frame1, frame2])
        cv2.imshow("video", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break