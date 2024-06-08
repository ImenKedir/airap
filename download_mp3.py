import os
from pytube import YouTube
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

def download_youtube_audio(url, folder, song_name):
    if folder == "i":
        output_folder = "/Users/imenkedir/dev/airap/data/instrumental"
        song_name = song_name + " - Instrumental.mp3"
    elif folder == "c":
        output_folder = "/Users/imenkedir/dev/airap/data/complete"
        song_name = song_name + " - Complete.mp3"
    else:
        print(f"Invalid folder type: {folder}")
        return

    try:
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Generate a unique temporary filename
        temp_audio_filename = f'temp_audio_{uuid.uuid4()}.mp4'

        # Download the YouTube video
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_file = audio_stream.download(output_path=output_folder, filename=temp_audio_filename)

        # Define the full path for the output file
        output_file_path = os.path.join(output_folder, song_name)

        # Convert the downloaded file to mp3
        AudioSegment.from_file(audio_file).export(output_file_path, format="mp3")

        # Remove the original file
        os.remove(audio_file)

        print(f"Downloaded and converted to MP3: {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_videos_concurrently(video_list):
    with ThreadPoolExecutor() as executor:
        future_to_video = {
            executor.submit(download_youtube_audio, video['url'], video['folder'], video['song_name']): video
            for video in video_list
        }

        for future in as_completed(future_to_video):
            video = future_to_video[future]
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred for video {video['url']}: {e}")

# Example usage
video_list = [
    {'url': "https://www.youtube.com/watch?v=BL8bIIbYlnU", 'folder': 'i', 'song_name': "106 BPM - Future - Shit"},
    {'url': "https://www.youtube.com/watch?v=tH0oqMxN3aQ", 'folder': 'c', 'song_name': "106 BPM - Future - Shit"},
    {'url': "https://www.youtube.com/watch?v=5tD9tnE2qx8", 'folder': 'i', 'song_name': "108 BPM - Lloyd Banks - Start It Up (Feat. Kanye West, Swizz Beatz, Ryan Leslie, & Fabolous)"},
    {'url': "https://www.youtube.com/watch?v=7XL84zQZ1nw", 'folder': 'c', 'song_name': "108 BPM - Lloyd Banks - Start It Up (Feat. Kanye West, Swizz Beatz, Ryan Leslie, & Fabolous)"},
    {'url': "https://www.youtube.com/watch?v=PfSQxhQjGGo", 'folder': 'i', 'song_name': "109 BPM - Fabolous - Young'n"},
    {'url': "https://www.youtube.com/watch?v=lsgPzHqxu_4", 'folder': 'c', 'song_name': "109 BPM - Fabolous - Young'n"},
    {'url': "https://www.youtube.com/watch?v=i2_kPKLJkBY", 'folder': 'i', 'song_name': "112 BPM - Eminem - Without Me"},
    {'url': "https://www.youtube.com/watch?v=-8xhmV3JoG4", 'folder': 'c', 'song_name': "112 BPM - Eminem - Without Me"},
    {'url': "https://www.youtube.com/watch?v=5Dff32msaXs", 'folder': 'i', 'song_name': "121 BPM - T-Pain - Can't Believe It (Feat. Lil Wayne)"},
    {'url': "https://www.youtube.com/watch?v=_FGR9xp0jHA", 'folder': 'c', 'song_name': "121 BPM - T-Pain - Can't Believe It (Feat. Lil Wayne)"},
    {'url': "https://www.youtube.com/watch?v=wHYEARyNAKU", 'folder': 'i', 'song_name': "123 BPM - Post Malone - Congratulations (Feat. Quavo)"},
    {'url': "https://www.youtube.com/watch?v=R8vpQdZErbw", 'folder': 'c', 'song_name': "123 BPM - Post Malone - Congratulations (Feat. Quavo)"},
    {'url': "https://www.youtube.com/watch?v=xKvO_UPCKkI", 'folder': 'i', 'song_name': "123 BPM - Soulja Boy Tell' Em - Pretty Boy Swag"},
    {'url': "https://www.youtube.com/watch?v=CbNwDo9wXCI", 'folder': 'c', 'song_name': "123 BPM - Soulja Boy Tell' Em - Pretty Boy Swag"},
    {'url': "https://www.youtube.com/watch?v=RQoP9rJsznE", 'folder': 'i', 'song_name': "125 BPM - Future - Covered N Money"},
    {'url': "https://www.youtube.com/watch?v=8qj2rrxrC4I", 'folder': 'c', 'song_name': "125 BPM - Future - Covered N Money"},
    {'url': "https://www.youtube.com/watch?v=xEKfw3qOygA", 'folder': 'i', 'song_name': "131 BPM - Future - Move That Dope (Feat. Pharrell, Pusha T & Casino)"},
    {'url': "https://www.youtube.com/watch?v=un92EJx33YQ", 'folder': 'c', 'song_name': "131 BPM - Future - Move That Dope (Feat. Pharrell, Pusha T & Casino)"},
    {'url': "https://www.youtube.com/watch?v=Rj3sKzfJtJM", 'folder': 'i', 'song_name': "140 BPM - Danny Brown - Lie4"},
    {'url': "https://www.youtube.com/watch?v=1Ty1rxfCg-w", 'folder': 'c', 'song_name': "140 BPM - Danny Brown - Lie4"},
    {'url': "https://www.youtube.com/watch?v=opye45UsfuI", 'folder': 'i', 'song_name': "140 BPM - Megan Thee Stallion - Sex Talk"},
    {'url': "https://www.youtube.com/watch?v=2YURpe_MhhQ", 'folder': 'c', 'song_name': "140 BPM - Megan Thee Stallion - Sex Talk"},
    {'url': "https://www.youtube.com/watch?v=GA-AKJ_ANJc", 'folder': 'i', 'song_name': "140 BPM - Soulja Boy Tell' Em - Crank Dat"},
    {'url': "https://www.youtube.com/watch?v=gvI8XJofCRc", 'folder': 'c', 'song_name': "140 BPM - Soulja Boy Tell' Em - Crank Dat"}
]


download_videos_concurrently(video_list)



