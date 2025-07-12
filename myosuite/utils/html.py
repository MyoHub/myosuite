from base64 import b64encode

from IPython.display import HTML


def show_video(video_path, video_width=400):

    video_file = open(video_path, "r+b").read()

    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(
        f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>"""
    )
