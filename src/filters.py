# src/filters.py
"""
Các hàm filter dữ liệu theo injury, equipment, level
"""

import pandas as pd
from .rules import INJURY_MAP


def filter_by_injury(df, injury):
    """
    Lọc bỏ các bài tập không phù hợp với chấn thương
    
    Args:
        df (pd.DataFrame): DataFrame chứa danh sách bài tập
        injury (str): Loại chấn thương (spine, shoulder, knee, wrist, neck)
        
    Returns:
        pd.DataFrame: DataFrame đã được lọc
    """
    if df is None or len(df) == 0:
        return df
    
    injury = injury.lower()
    
    if injury not in INJURY_MAP:
        print(f"⚠️ Không tìm thấy thông tin về chấn thương '{injury}'")
        return df
    
    injury_rules = INJURY_MAP[injury]
    
    # Lọc bỏ các bài tập không an toàn
    filtered_df = df[
        ~df["BodyPart"].str.lower().isin(injury_rules["bodypart"]) &
        ~df["Equipment"].str.lower().isin(injury_rules["equipment"]) &
        ~df["Type"].str.lower().isin(injury_rules["type"])
    ]
    
    if len(filtered_df) == 0:
        print(f"⚠️ Không còn bài tập nào phù hợp sau khi lọc chấn thương '{injury}'")
        return None
    
    return filtered_df


def filter_by_equipment(df, equipment):
    """
    Lọc bài tập theo thiết bị
    
    Args:
        df (pd.DataFrame): DataFrame chứa danh sách bài tập
        equipment (str): Loại thiết bị
        
    Returns:
        pd.DataFrame: DataFrame đã được lọc
    """
    if df is None or len(df) == 0 or not equipment:
        return df
    
    filtered_df = df[df["Equipment"].str.lower() == equipment.lower()]
    
    if len(filtered_df) == 0:
        print(f"⚠️ Không tìm thấy bài tập nào với thiết bị '{equipment}'")
        return None
    
    return filtered_df


def filter_by_level(df, level):
    """
    Lọc bài tập theo cấp độ
    
    Args:
        df (pd.DataFrame): DataFrame chứa danh sách bài tập
        level (str): Cấp độ (Beginner, Intermediate, Advanced)
        
    Returns:
        pd.DataFrame: DataFrame đã được lọc
    """
    if df is None or len(df) == 0 or not level:
        return df
    
    filtered_df = df[df["Level"].str.lower() == level.lower()]
    
    if len(filtered_df) == 0:
        print(f"⚠️ Không tìm thấy bài tập nào với cấp độ '{level}'")
        return None
    
    return filtered_df


def filter_by_type(df, type_):
    """
    Lọc bài tập theo loại
    
    Args:
        df (pd.DataFrame): DataFrame chứa danh sách bài tập
        type_ (str): Loại bài tập (Strength, Cardio, etc.)
        
    Returns:
        pd.DataFrame: DataFrame đã được lọc
    """
    if df is None or len(df) == 0 or not type_:
        return df
    
    filtered_df = df[df["Type"].str.lower() == type_.lower()]
    
    if len(filtered_df) == 0:
        print(f"⚠️ Không tìm thấy bài tập nào với loại '{type_}'")
        return None
    
    return filtered_df


def apply_all_filters(df, injury=None, equipment=None, level=None, type_=None):
    """
    Áp dụng tất cả các filter
    
    Args:
        df (pd.DataFrame): DataFrame chứa danh sách bài tập
        injury (str, optional): Loại chấn thương
        equipment (str, optional): Loại thiết bị
        level (str, optional): Cấp độ
        type_ (str, optional): Loại bài tập
        
    Returns:
        pd.DataFrame: DataFrame đã được lọc
    """
    if df is None or len(df) == 0:
        return df
    
    # Áp dụng từng filter
    if injury:
        df = filter_by_injury(df, injury)
        if df is None or len(df) == 0:
            return None
    
    if equipment:
        df = filter_by_equipment(df, equipment)
        if df is None or len(df) == 0:
            return None
    
    if level:
        df = filter_by_level(df, level)
        if df is None or len(df) == 0:
            return None
    
    if type_:
        df = filter_by_type(df, type_)
        if df is None or len(df) == 0:
            return None
    
    return df