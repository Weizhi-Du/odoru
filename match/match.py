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
    # action 原点
    body1, body2 = [], []
    for i in [0, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28]:
        if action[i].visibility < 0.5 or target[i].visibility < 0.5: continue
        body1.append([action[i].x * w , action[i].y * h])
        body2.append([target[i].x * w, target[i].y * h])
    _, _, disparity = procrustes(np.array(body1), np.array(body2))
    return disparity * 1e2


def match_batch(action, target, w, h):
    best = 1e9
    for i in action:
        for j in target:
            best = min(best, match(i, j, w, h))
    return best


# main
if __name__ == '__main__':
    action = read_video('weizhi.mp4')
    target = read_video('weien.mp4')[30:]
    w = int(target[0].shape[1])
    h = int(target[0].shape[0])
    for i in range(max(len(action), len(target))):
        frame1, frame2 = cv2.resize(action[-1], (w//2, h//2)), cv2.resize(target[-1], (w//2, h//2))
        if i < len(action): frame1 = cv2.resize(action[i], (w//2, h//2))
        if i < len(target): frame2 = cv2.resize(target[i], (w//2, h//2))
        cv2.imshow("video", cv2.hconcat([frame1, frame2]))
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break