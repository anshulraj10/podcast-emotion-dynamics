import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
INPUT_DIR = Path(__file__).parent.parent / Path("data/processed/emotions")
OUTPUT_FILE = Path(__file__).parent.parent / Path("data/processed/emotions.csv")

def merge_emotions(input_dir) -> pd.DataFrame:
    """ Merge all the csv files of emotions into one csv file

    Args:
        input_dir (str): The directory where all csv files are stored.

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    all_files = list(input_dir.glob("*.csv"))
    df_list = []

    for file in tqdm(all_files, desc="Merging emotion files"):
        df = pd.read_csv(file)
        video_id = file.stem.replace('_emotions', '')
        df['video_id'] = video_id
        df_list.append(df)

    emotions_df = pd.concat(df_list, ignore_index=True)
    # emotions_df.to_csv(OUTPUT_FILE, index=False)
    return emotions_df

def resize_list_with_interpolation(original_list, target_size=100):
    """
    Resizes a list to a fixed target_size using linear interpolation.

    Args:
        original_list (list): The input list of numerical values.
        target_size (int): The desired fixed size of the output list.

    Returns:
        list: A new list of the target_size with interpolated values.
    """

    original_indices = np.linspace(0, len(original_list) - 1, num=len(original_list))
    target_indices = np.linspace(0, len(original_list) - 1, num=target_size)

    interpolated_values = np.interp(target_indices, original_indices, original_list)
    return interpolated_values.tolist()

def process_emotions(resize_list_with_interpolation, emotions_df):
    """
    Main function to process the emotions and resize each emotion list to a fixed size

    Args:
        resize_list_with_interpolation (func): Function to resize the list
        emotions_df (pd.DataFrame): The whole emotions df

    Returns:
        list: list of all processed emotions
    """
    emotions = []
    
    video_ids = emotions_df['video_id'].unique()
    EMOTION_COLUMNS = [col for col in emotions_df.columns if col not in ['start', 'end', 'text', 'time', 'video_id']]
    
    for video_id in tqdm(video_ids, desc="Processing emotion files"):
        video_emotions_df = emotions_df[emotions_df['video_id'] == video_id]
        
        if len(video_emotions_df) < 42:
            continue
        
        data = {"video_id": video_id,
                **{ emotion: resize_list_with_interpolation(video_emotions_df[emotion].values)
                   for emotion in EMOTION_COLUMNS
                   }
                }
        
        emotions.append(data)
    return emotions

if __name__ == "__main__":
    emotions_df = merge_emotions(INPUT_DIR)
    
    emotions = process_emotions(resize_list_with_interpolation, emotions_df)
    
    emotions_df = pd.DataFrame(emotions)
    emotions_df.to_csv(OUTPUT_FILE, index=False)