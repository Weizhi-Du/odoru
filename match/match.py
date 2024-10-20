import cv2
import mediapipe as mp
from tqdm import tqdm
import numpy as np
from scipy.spatial import procrustes


# 模型配置
mp_pose = mp.solutions.pose
pose_landmark = mp_pose.PoseLandmark
mp_drawing = mp.solutions.drawing_utils


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


# 动作识别
def detect_pose(model, frame):
    res = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return res


# 动作匹配 （一帧）
def match(action, target, w, h):
    if action.pose_landmarks is None or target.pose_landmarks is None: return 100
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


# 批量匹配
def match_batch(action, target, w, h):
    best = 1e9
    for i in action:
        for j in target:
            best = min(best, match(i, j, w, h))
    return best


def draw(frame, result):
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)


# main
def compare():
    action = read_video('./match/weizhi.mp4')
    target = read_video('./match/weien.mp4')[30:]
    w, h = 270, 480
    batch_action, batch_target = [], []
    score = 1
    model1, model2 = mp_pose.Pose(), mp_pose.Pose()
    for i in range(max(len(action), len(target))):
        frame1, frame2 = cv2.resize(action[-1], (w, h)), cv2.resize(target[-1], (w, h))
        if i < len(action): frame1 = cv2.resize(action[i], (w, h))
        if i < len(target): frame2 = cv2.resize(target[i], (w, h))

        res1, res2 = detect_pose(model1, frame1), detect_pose(model2, frame2)
        batch_action.append(res1)
        batch_target.append(res2)
        if len(batch_target) == 5:
            score = match_batch(batch_action, batch_target, w, h)
            batch_action, batch_target = [], []

        draw(frame1, res1); draw(frame2, res2)
        frame = cv2.hconcat([frame1, frame2])
        cv2.putText(frame, str(round(score,1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == '__main__':
    target = read_video('./match/weien.mp4')
    cap = cv2.VideoCapture(0)
    w, h = 270, 480

    output_file = "/match/pose_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, (w*2, h))

    score, i = 0, 0
    batch_action, batch_target = [], []
    model1, model2 = mp_pose.Pose(), mp_pose.Pose()
    while cap.isOpened():
        ret, frame1 = cap.read()
        if not ret: break
        frame1, frame2 = cv2.flip(frame1[-h:, -w:], 1), cv2.resize(target[-1], (w, h))
        if i < len(target): frame2 = cv2.resize(target[i], (w, h))
        
        res1, res2 = detect_pose(model1, frame1), detect_pose(model2, frame2)
        batch_action.append(res1); batch_target.append(res2)
        if len(batch_target) == 10:
            score = match_batch(batch_action, batch_target, w, h)
            batch_action, batch_target = [], []

        draw(frame1, res1); draw(frame2, res2)
        frame = cv2.hconcat([frame1, frame2])
        cv2.putText(frame, str(round(score,1)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        out.write(frame)

        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        i += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()