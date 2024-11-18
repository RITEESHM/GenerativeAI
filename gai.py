import instaloader
from moviepy.editor import VideoFileClip
import os
import logging
import shutil
import requests
from bs4 import BeautifulSoup
import openai
import librosa
from pydub import AudioSegment

class InstagramScraper:
    def __init__(self):
        self.loader = instaloader.Instaloader(download_videos=True)
        self.temp_dir = "temp_downloads"
        os.makedirs(self.temp_dir, exist_ok=True)

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate the user using provided Instagram credentials."""
        try:
            self.loader.login(username, password)
            return True
        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            return False

    def fetch_creator_content(self, username: str, max_posts: int = 10):
        """Fetch content from a given Instagram creator's profile."""
        try:
            profile = instaloader.Profile.from_username(self.loader.context, username)
            posts = profile.get_posts()
            videos, captions = [], []

            for post in posts:
                if len(videos) >= max_posts:
                    break
                if post.is_video:
                    video_data = self._process_video(post)
                    if video_data:
                        videos.append(video_data)
                        captions.append(post.caption or "")
            
            self._cleanup_temp_files()  
            return {'videos': videos, 'captions': captions}
        except Exception as e:
            logging.error(f"Content fetching failed: {e}")
            return None

    def _process_video(self, post):
        """Process the video by downloading and extracting audio."""
        try:
            video_path = f"{self.temp_dir}/{post.shortcode}.mp4"
            self.loader.download_post(post, target=self.temp_dir)

            with VideoFileClip(video_path) as video:
                audio_path = self._extract_audio(video, post.shortcode)
                return {
                    'duration': video.duration,
                    'audio_path': audio_path
                }
        except Exception as e:
            logging.error(f"Video processing failed for {post.shortcode}: {e}")
            return None

    def _extract_audio(self, video, shortcode):
        audio_path = f"{self.temp_dir}/audio_{shortcode}.mp3"
        try:
            video.audio.write_audiofile(audio_path)
            return audio_path
        except Exception as e:
            logging.error(f"Audio extraction failed for {shortcode}: {e}")
            return None

    def _cleanup_temp_files(self):
        try:
            shutil.rmtree(self.temp_dir)
            logging.info(f"Temporary files cleaned up from {self.temp_dir}.")
        except Exception as e:
            logging.error(f"Failed to clean up temporary files: {e}")

    def analyze_creator_style(self, captions):
        creator_style = "a specific writing style identified from captions"
        return creator_style

    def fetch_product_details(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('h1').text
            description = soup.find('div', {'class': 'product-description'}).text
            return title, description
        except Exception as e:
            logging.error(f"Failed to fetch product details: {e}")
            return None, None

    def generate_review(self, title, description, creator_style):
        prompt = f"Write a product review in the style of {creator_style} for the product titled '{title}' with the following description: {description}"
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=150)
        return response.choices[0].text.strip()

    def generate_video_script(self, review, creator_style):
        prompt = f"Create a video script in the style of {creator_style} for the following product review: {review}"
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=300)
        return response.choices[0].text.strip()

    def synthesize_voice(self, script, voice_model_path):
        tts_model = load_trained_tts_model(voice_model_path)
        voice_clip = tts_model.synthesize(script)
        return voice_clip


def main():
    scraper = InstagramScraper()
    if not scraper.authenticate('your_username', 'your_password'):
        print("Authentication failed.")
        return

    # Fetch creator's content
    creator_content = scraper.fetch_creator_content('creator_username')
    if not creator_content:
        print("Failed to fetch creator content.")
        return

    # Analyze creator's style
    creator_style = scraper.analyze_creator_style(creator_content['captions'])

    # Fetch product details
    product_url = 'https://example.com/product-page'
    title, description = scraper.fetch_product_details(product_url)
    if not title or not description:
        print("Failed to fetch product details.")
        return

    # Generate product review
    review = scraper.generate_review(title, description, creator_style)

    # Generate video script
    script = scraper.generate_video_script(review, creator_style)

    # Synthesize voice clip
    voice_model_path = 'path/to/creator_voice_model'
    voice_clip = scraper.synthesize_voice(script, voice_model_path)

    # Save or use the voice clip
    with open('output_voice_clip.wav', 'wb') as f:
        f.write(voice_clip)

if __name__ == "__main__":
    main()
