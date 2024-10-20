import pygame
import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import procrustes

pygame.init()

# Set up the display
window_width, window_height = 960, 540
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Dance Game")
clock = pygame.time.Clock()

# PoseMatcher class (from match.py)
class PoseMatcher:
    def __init__(self, w=270, h=480, max_step=10):
        self.w, self.h = w, h
        self.max_step = max_step
        self.batch_action, self.batch_target = [], []
        self.score = 0
        # Model configuration
        self.mp_pose = mp.solutions.pose
        self.pose_landmark = self.mp_pose.PoseLandmark
        self.mp_drawing = mp.solutions.drawing_utils
        self.model1, self.model2 = self.mp_pose.Pose(), self.mp_pose.Pose()

    # Pose detection
    def detect_pose(self, model, frame):
        res = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return res

    # Single frame matching
    def match(self, action, target):
        if action.pose_landmarks is None or target.pose_landmarks is None:
            return 100
        action = action.pose_landmarks.landmark
        target = target.pose_landmarks.landmark
        body1, body2 = [], []
        for i in [0, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26, 27, 28]:
            if action[i].visibility < 0.5 or target[i].visibility < 0.5:
                continue
            body1.append([action[i].x * self.w, action[i].y * self.h])
            body2.append([target[i].x * self.w, target[i].y * self.h])
        if not body1 or not body2:
            return 100
        _, _, disparity = procrustes(np.array(body1), np.array(body2))
        return disparity * 1e2

    # Batch matching
    def match_batch(self, action, target):
        best = 1e9
        for i in action:
            for j in target:
                best = min(best, self.match(i, j))
        return best

    # Draw skeleton
    def draw(self, frame, result):
        if result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

    # Call function
    def __call__(self, frame1, frame2):
        res1 = self.detect_pose(self.model1, frame1)
        res2 = self.detect_pose(self.model2, frame2)
        self.batch_action.append(res1)
        self.batch_target.append(res2)
        if len(self.batch_target) == self.max_step:
            self.score = self.match_batch(self.batch_action, self.batch_target)
            self.batch_action, self.batch_target = [], []
        self.draw(frame1, res1)
        self.draw(frame2, res2)
        return frame1, frame2, self.score

def video_selection_screen():
    # Get the list of video files
    video_folder = 'videos'  # Adjust the folder name
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

    # Generate thumbnails for each video
    thumbnails = []
    for video_file in video_files:
        cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
        ret, frame = cap.read()
        cap.release()
        if ret:
            # Resize frame to thumbnail size
            frame = cv2.resize(frame, (160, 90))
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to Pygame surface
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
            thumbnails.append((video_file, frame_surface))
        else:
            thumbnails.append((video_file, None))  # No thumbnail

    selected_video = None

    # Display the thumbnails
    while not selected_video:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                # Check if any thumbnail is clicked
                for idx, (video_file, thumbnail) in enumerate(thumbnails):
                    x = (idx % 5) * 170 + 20  # Adjust positions
                    y = (idx // 5) * 100 + 20
                    rect = pygame.Rect(x, y, 160, 90)
                    if rect.collidepoint(pos):
                        selected_video = os.path.join(video_folder, video_file)
                        break

        screen.fill((0, 0, 0))
        # Draw thumbnails
        for idx, (video_file, thumbnail) in enumerate(thumbnails):
            x = (idx % 5) * 170 + 20  # Adjust positions
            y = (idx // 5) * 100 + 20
            if thumbnail:
                screen.blit(thumbnail, (x, y))
            else:
                # Draw placeholder rectangle
                pygame.draw.rect(screen, (255, 255, 255), (x, y, 160, 90))
            # Draw video filename
            font = pygame.font.Font(None, 24)
            text_surface = font.render(video_file, True, (255, 255, 255))
            screen.blit(text_surface, (x, y + 90))

        pygame.display.flip()
        clock.tick(30)

    return selected_video

def countdown_screen():
    countdown_numbers = ['3', '2', '1']
    font = pygame.font.Font(None, 144)
    for number in countdown_numbers:
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < 1000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            screen.fill((0, 0, 0))
            text_surface = font.render(number, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(window_width//2, window_height//2))
            screen.blit(text_surface, text_rect)
            pygame.display.flip()
            clock.tick(30)

def main_game(selected_video):
    # Initialize video capture
    video_cap = cv2.VideoCapture(selected_video)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize webcam capture
    webcam_cap = cv2.VideoCapture(0)
    webcam_width = int(webcam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(webcam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set display size
    display_width = window_width // 2
    display_height = window_height

    # Initialize PoseMatcher
    w, h = display_width, display_height
    model = PoseMatcher(w, h)
    score = 0
    scores = []  # Store scores to update progress bar

    frame_count = 0

    running = True
    while running:
        frame_count += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ret_video, frame_video = video_cap.read()
        ret_webcam, frame_webcam = webcam_cap.read()
        if not ret_video or not ret_webcam:
            break

        # Resize frames to display size
        frame_video = cv2.resize(frame_video, (w, h))
        frame_webcam = cv2.flip(frame_webcam, 1)
        frame_webcam = cv2.resize(frame_webcam, (w, h))

        # Process frames with PoseMatcher
        frame_video, frame_webcam, current_score = model(frame_video, frame_webcam)

        # Every 15 frames (0.5 seconds), update the score
        if frame_count % 15 == 0:
            score = current_score
            scores.append(score)

        # Convert frames to RGB
        frame_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)
        frame_webcam = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)

        # Convert frames to Pygame surfaces
        surface_video = pygame.surfarray.make_surface(np.rot90(frame_video))
        surface_webcam = pygame.surfarray.make_surface(np.rot90(frame_webcam))

        # Display frames
        screen.blit(surface_video, (0, 0))
        screen.blit(surface_webcam, (display_width, 0))

        # Update progress bar
        progress_bar_width = window_width
        progress_bar_height = 20
        progress = frame_count / total_frames
        progress_width = int(progress * progress_bar_width)

        # Draw progress bar background
        pygame.draw.rect(screen, (50, 50, 50), (0, window_height - progress_bar_height, progress_bar_width, progress_bar_height))

        # Determine color based on last score
        if scores:
            last_score = scores[-1]
            if last_score < 1.0:
                color = (255, 165, 0)  # Orange
            elif 1.0 <= last_score < 1.5:
                color = (255, 192, 203)  # Pink
            elif 1.5 <= last_score < 2.5:
                color = (0, 255, 0)  # Green
            else:
                color = (128, 128, 128)  # Gray
        else:
            color = (255, 255, 255)

        # Draw progress bar
        pygame.draw.rect(screen, color, (0, window_height - progress_bar_height, progress_width, progress_bar_height))

        # Display score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {round(score, 1)}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(30)  # Limit to 30 FPS

    # Release video captures
    video_cap.release()
    webcam_cap.release()

    # At the end, display performance screen
    performance_screen(scores)

def performance_screen(scores):
    # Calculate average score
    avg_score = sum(scores) / len(scores) if scores else 0

    # Determine performance based on average score
    if avg_score < 1.0:
        performance = "Perfect!"
    elif 1.0 <= avg_score < 1.5:
        performance = "Great!"
    elif 1.5 <= avg_score < 2.5:
        performance = "Good"
    else:
        performance = "Keep Practicing!"

    # Display performance
    font = pygame.font.Font(None, 72)
    text_surface = font.render(performance, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(window_width//2, window_height//2))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))
        screen.blit(text_surface, text_rect)
        pygame.display.flip()
        clock.tick(30)

if __name__ == '__main__':
    selected_video = video_selection_screen()
    countdown_screen()
    main_game(selected_video)
    pygame.quit()