from base64 import b64encode

from IPython.display import HTML


def show_video(video_path, video_width=400):
    """Display a video file in Jupyter notebook using HTML5 video element.

    This function reads a video file from the given path, encodes it as base64,
    and returns an HTML object that can be displayed in a Jupyter notebook.
    The video will autoplay and include standard HTML5 video controls.

    Args:
        video_path (str): Path to the video file to be displayed.
        video_width (int, optional): Width of the video in pixels.
            Defaults to 400.

    Returns:
        IPython.display.HTML: HTML object containing the video element that
            can be displayed in a Jupyter notebook.

    Raises:
        FileNotFoundError: If the video file at the specified path does not
            exist.
        IOError: If there are issues reading the video file.

    Example:
        >>> from myosuite.utils.html import show_video
        >>> video_html = show_video("path/to/video.mp4", video_width=600)
        >>> display(video_html)
    """
    video_file = open(video_path, "r+b").read()

    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    html_content = (
        f"""<video autoplay width={video_width} controls>"""
        f"""<source src="{video_url}"></video>"""
    )
    return HTML(html_content)
