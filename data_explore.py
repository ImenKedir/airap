import os
import torchaudio
import torchaudio.transforms as T

def get_mp3_duration(mp3_path):
    waveform, sample_rate = torchaudio.load(mp3_path)
    duration = waveform.shape[1] / sample_rate
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    return minutes, seconds, duration

def print_song_durations(folder_paths):
    total_duration = 0

    for folder_path in folder_paths:
        print(f"Processing folder: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith('.mp3'):
                mp3_path = os.path.join(folder_path, filename)
                minutes, seconds, duration = get_mp3_duration(mp3_path)
                print(f"{filename}: {minutes}:{seconds:02d}")
                total_duration += duration
    
    total_minutes = int(total_duration // 60)
    total_seconds = int(total_duration % 60)
    print(f"\nTotal time: {total_minutes}:{total_seconds:02d}")

# Example usage:
folder_paths = [
    '/Users/imenkedir/Desktop/music/acapellas/Juesswork - Acapella Stash Vol. 1',
    '/Users/imenkedir/Desktop/music/acapellas/Juesswork - Acapella Stash Vol. 2',
    '/Users/imenkedir/Desktop/music/acapellas/Juesswork - Acapella Stash Vol. 3'
]  # Replace with the paths to your MP3 folders
print_song_durations(folder_paths)
