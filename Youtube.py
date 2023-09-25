import yt_dlp

url = input("Enter Video url: ")

saving_location = input("Enter saving location (e.g. ~/path/): ")

ydl_opts = {
    'outtmpl': f'~/{saving_location} + %(title)s.%(ext)s'
    }

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print(f"Video saved to {saving_location}")
