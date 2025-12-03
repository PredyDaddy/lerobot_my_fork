import pygame
import time
import sys

def play_audio(file_path):
    """使用 pygame 播放音频文件"""

    # 初始化 pygame 混音器
    pygame.mixer.init()

    try:
        # 加载音频文件
        print(f"正在加载音频文件: {file_path}")
        pygame.mixer.music.load(file_path)

        # 播放音频
        print("开始播放音频...")
        pygame.mixer.music.play()

        # 等待音频播放完成
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        print("音频播放完成")

    except Exception as e:
        print(f"播放音频时出错: {e}")
    finally:
        # 退出 pygame
        pygame.mixer.quit()
        pygame.quit()

if __name__ == "__main__":
    # 音频文件路径
    audio_file = "11-2.wav"

    # 播放音频
    play_audio(audio_file)
