import pygame
import os
import cv2
import threading
import time
import random

# Initialize Pygame
pygame.init()

# Constants for the UI
WIDTH, HEIGHT = 1280, 720
BG_COLOR = (30, 30, 30)
COUNTDOWN_COLORS = [(255, 100, 100), (255, 200, 100), (100, 255, 100)]
COUNTDOWN_NUMBERS = [3, 2, 1]
FONT = pygame.font.Font(None, 150)

# Paths
VIDEO_FOLDER = './videos'  # Assume a folder where videos are stored
THUMBNAIL_SIZE = (160, 90)

# Create the main window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dance Game")

# Video Display Threading Setup
video_selected = None
webcam_stream = None
game_running = False
similarity_score = None
progress_bar_color = (255, 255, 255)

# Load video thumbnails from folder
def load_video_thumbnails(folder):
    videos = []
    for filename in os.listdir(folder):
        if filename.endswith(".mp4"):
            filepath = os.path.join(folder, filename)
            videos.append(filepath)
    return videos

# Draw the countdown
def draw_countdown(number):
    screen.fill(BG_COLOR)
    text = FONT.render(str(number), True, COUNTDOWN_COLORS[number - 1])
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    pygame.time.delay(1000)

# Game Main Loop
def game_main_loop(video_path):
    global game_running, webcam_stream, similarity_score, progress_bar_color
    
    # Load the video and prepare webcam
    video = cv2.VideoCapture(video_path)
    webcam = cv2.VideoCapture(0)  # Assume the webcam is index 0
    
    # Game loop
    start_time = time.time()
    while game_running:
        screen.fill(BG_COLOR)
        
        # Read a frame from the video
        ret, video_frame = video.read()
        if not ret:
            break
        
        # Read a frame from the webcam
        ret, webcam_frame = webcam.read()
        if not ret:
            break
        
        # Simulate score computation from the provided skeleton-detecting code (use random for now)
        similarity_score = random.uniform(0, 3)
        
        # Update progress bar color based on score
        if similarity_score < 1.0:
            progress_bar_color = (255, 165, 0)  # Orange for perfect
        elif similarity_score < 1.5:
            progress_bar_color = (255, 105, 180)  # Pink for great
        elif similarity_score < 2.5:
            progress_bar_color = (0, 255, 0)  # Green for good
        else:
            progress_bar_color = (169, 169, 169)  # Gray for bad
        
        # Draw the video on the left and webcam on the right
        video_surface = pygame.surfarray.make_surface(cv2.resize(video_frame, (WIDTH//2, HEIGHT)))
        webcam_surface = pygame.surfarray.make_surface(cv2.resize(webcam_frame, (WIDTH//2, HEIGHT)))
        screen.blit(video_surface, (0, 0))
        screen.blit(webcam_surface, (WIDTH//2, 0))
        
        # Draw progress bar
        pygame.draw.rect(screen, progress_bar_color, (0, HEIGHT-30, WIDTH, 30))
        
        pygame.display.flip()
        pygame.time.delay(33)  # Delay for ~30fps

    # Cleanup
    video.release()
    webcam.release()
    pygame.quit()

# Main Program
def main():
    global video_selected, game_running
    
    clock = pygame.time.Clock()
    running = True
    
    # Load video thumbnails
    video_paths = load_video_thumbnails(VIDEO_FOLDER)
    thumbnails = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            thumbnail = cv2.resize(frame, THUMBNAIL_SIZE)
            thumbnail_surface = pygame.surfarray.make_surface(thumbnail)
            thumbnails.append((thumbnail_surface, video_path))
        cap.release()
    
    # Main menu loop
    while running:
        screen.fill(BG_COLOR)
        
        # Display video thumbnails
        for i, (thumbnail_surface, video_path) in enumerate(thumbnails):
            x = (i % 5) * (THUMBNAIL_SIZE[0] + 20) + 20
            y = (i // 5) * (THUMBNAIL_SIZE[1] + 20) + 20
            screen.blit(thumbnail_surface, (x, y))
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                for i, (thumbnail_surface, video_path) in enumerate(thumbnails):
                    x = (i % 5) * (THUMBNAIL_SIZE[0] + 20) + 20
                    y = (i // 5) * (THUMBNAIL_SIZE[1] + 20) + 20
                    if x <= mouse_x <= x + THUMBNAIL_SIZE[0] and y <= mouse_y <= y + THUMBNAIL_SIZE[1]:
                        video_selected = video_path
                        print(f"Video {video_path} selected!")
                        break
        
        pygame.display.flip()
        clock.tick(30)
        
        # If a video is selected, start the game
        if video_selected:
            # Countdown before starting
            for number in COUNTDOWN_NUMBERS:
                draw_countdown(number)
            
            game_running = True
            game_thread = threading.Thread(target=game_main_loop, args=(video_selected,))
            game_thread.start()
            game_thread.join()  # Wait for the game to finish
            running = False

    pygame.quit()

# Run the program
main()
