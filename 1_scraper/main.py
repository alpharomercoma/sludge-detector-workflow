from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
import os
import time
import json
from urllib.parse import urlparse
import sys
import yt_dlp
import subprocess

def identify_platform(url):
    """
    Identify the platform and return appropriate selector
    """
    domain = urlparse(url).netloc.lower()
    if any(x in domain for x in ['youtube.com', 'youtu.be']):
        return 'youtube', "a#video-title-link, a[href*='/shorts/']"
    elif 'instagram.com' in domain:
        return 'instagram', "a[href*='/reel/'], a[href*='/p/']"
    elif 'tiktok.com' in domain:
        return 'tiktok', "a[href*='/video/']"
    else:
        return None, None

def extract_video_id(url, platform):
    """
    Extract video ID from various platforms
    """
    try:
        if platform == "youtube":
            if "youtu.be" in url:
                return url.split('/')[-1].split('?')[0]
            if "watch?v=" in url:
                return url.split('watch?v=')[1].split('&')[0]
            if "shorts/" in url:
                return url.split('shorts/')[-1].split('?')[0]
        elif platform == "instagram":
            if '/reel/' in url:
                return url.split('/reel/')[-1].split('/')[0]
            else:
                return url.split('/p/')[-1].split('/')[0]
        elif platform == "tiktok":
            return url.split('/video/')[-1].split('?')[0]
        return None
    except:
        return None

def get_video_links(driver, url, selector):
    """
    Extract all video links from the page
    """
    print(f"Navigating to {url}")
    driver.get(url)
    time.sleep(35)  # Wait for page load

    # Scroll page
    print("Scrolling page...")
    scroll_pause_time = 1
    screen_height = driver.execute_script("return window.screen.height;")
    i = 1

    while i < 10:  # Limit scrolling to avoid too many requests
        driver.execute_script(f"window.scrollTo(0, {screen_height * i});")
        time.sleep(scroll_pause_time)
        i += 1

    # Extract links
    script = f"""
    let links = [];
    document.querySelectorAll("{selector}").forEach(anchor => {{
        links.push(anchor.href);
    }});
    return links;
    """
    video_links = driver.execute_script(script)
    return list(dict.fromkeys(video_links))  # Remove duplicates

def is_ffmpeg_installed():
    """
    Check if FFmpeg is installed on the system
    """
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def download_video(url, video_id):
    """
    Download video using yt-dlp for YouTube Shorts or sstik.io for TikTok
    """
    print(f"Downloading video {video_id} from: {url}")

    # Check if FFmpeg is installed
    if not is_ffmpeg_installed():
        print("FFmpeg is not installed. Please install FFmpeg to enable audio merging.")
        return False

    # Check if the video is a YouTube Shorts
    if "shorts/" in url:
        try:
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',  # Download the best video and audio
                'outtmpl': f'videos/{video_id}.mp4',   # Save to the specified path
                'postprocessors': [{                     # Merge video and audio
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',            # Output format
                }],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print("Download complete.")
            return True
        except Exception as e:
            print("An error occurred during download:", str(e))
            return False

    # Check if the video is from TikTok
    elif "tiktok.com" in url:
        try:
            # Use sstik.io to get the download link
            response = requests.get(f"https://sstik.io/?url={url}")
            response.raise_for_status()
            result = response.json()

            if 'url' in result:
                # Create directory for videos
                os.makedirs("videos", exist_ok=True)
                video_path = f"videos/{video_id}.mp4"

                # Download the TikTok video
                video_response = requests.get(result['url'], stream=True)
                video_response.raise_for_status()

                with open(video_path, "wb") as file:
                    for chunk in video_response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)

                print(f"Video saved at {video_path}")
                return True
            else:
                print(f"No download URL found in response: {result}")
                return False

        except Exception as e:
            print(f"Failed to download TikTok video {video_id}: {e}")
            return False

    print("Unsupported platform for download.")
    return False

def main():
    # Check if URL was provided as command line argument
    if len(sys.argv) != 2:
        print("Usage: python video_scraper.py <URL>")
        print("Example: python video_scraper.py https://www.tiktok.com/@username")
        return

    url = sys.argv[1]

    # Identify platform
    platform, selector = identify_platform(url)
    if not platform or not selector:
        print("Unsupported platform. Please enter a URL from YouTube, Instagram, or TikTok")
        return

    print(f"\nIdentified platform: {platform.upper()}")

    # Setup browser
    print("Starting browser...")
    options = Options()
    options.add_argument("start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        # Get video links
        video_links = get_video_links(driver, url, selector)

        if not video_links:
            print("No videos found!")
            return

        print(f"\nFound {len(video_links)} videos. Starting download...")

        # Download videos
        for video_url in video_links:
            video_id = extract_video_id(video_url, platform)
            if not video_id:
                print(f"Skipping video, could not extract ID from: {video_url}")
                continue

            # Add platform prefix to video ID
            video_id = f"{platform}_{video_id}"

            print(f"\nProcessing video {video_id}")
            success = download_video(video_url, video_id)

            if success:
                print("Download completed successfully!")
            else:
                print("Download failed.")

            # Wait between downloads
            time.sleep(5)

    finally:
        driver.quit()

if __name__ == "__main__":
    main()