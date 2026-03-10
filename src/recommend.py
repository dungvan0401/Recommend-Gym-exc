# src/recommend.py
"""
Các hàm gợi ý bài tập gym
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .models import get_model
from .rules import GOAL_MAP, DEFAULT_TOP_N, DEFAULT_SAMPLE_SIZE
from .filters import apply_all_filters


def recommend_from_description(user_input, top_n=DEFAULT_TOP_N, injury=None, equipment=None, level=None):
    """
    Gợi ý bài tập dựa trên mô tả tự nhiên
    
    Args:
        user_input (str): Mô tả bài tập của người dùng
        top_n (int): Số lượng bài tập gợi ý
        injury (str, optional): Loại chấn thương cần tránh
        equipment (str, optional): Thiết bị
        level (str, optional): Cấp độ
        
    Returns:
        pd.DataFrame: Danh sách bài tập được gợi ý
    """
    model = get_model()
    data = model.get_data()
    bert_embeddings = model.get_embeddings()
    
    # Tạo vector cho input của user
    user_vector = model.get_bert_vector_for_text(user_input)
    
    # Tính similarity scores
    sim_scores = cosine_similarity(user_vector, bert_embeddings)[0]
    
    # Tạo DataFrame kết quả
    results = pd.DataFrame({
        'Title': data['Title'],
        'Type': data['Type'],
        'BodyPart': data['BodyPart'],
        'Equipment': data['Equipment'],
        'Level': data['Level'],
        'Desc_vi': data['Desc_vi'],
        'Similarity': sim_scores
    })
    
    # Sắp xếp theo similarity
    results = results.sort_values(by='Similarity', ascending=False)
    
    # Áp dụng filters
    results = apply_all_filters(results, injury=injury, equipment=equipment, level=level)
    
    if results is None or len(results) == 0:
        return None
    
    return results.head(top_n).reset_index(drop=True)


def recommend_by_title(title, top_n=DEFAULT_TOP_N, type_=None, bodypart=None, equipment=None, level=None, injury=None):
    """
    Gợi ý bài tập tương tự dựa trên tên bài tập
    
    Args:
        title (str): Tên bài tập gốc
        top_n (int): Số lượng bài tập gợi ý
        type_ (str, optional): Loại bài tập
        bodypart (str, optional): Nhóm cơ
        equipment (str, optional): Thiết bị
        level (str, optional): Cấp độ
        injury (str, optional): Loại chấn thương
        
    Returns:
        pd.DataFrame: Danh sách bài tập được gợi ý
    """
    model = get_model()
    data = model.get_data()
    cosine_sim = model.get_cosine_sim()
    
    # Kiểm tra title có tồn tại không
    if title not in data['Title'].values:
        print(f" Không tìm thấy bài tập '{title}' trong dữ liệu.")
        return None
    
    # Lấy index của bài tập
    idx = data.index[data['Title'] == title][0]
    
    # Tính similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+50]  # Bỏ chính nó, lấy nhiều hơn để filter
    
    # Tạo DataFrame kết quả
    similar_exercises = pd.DataFrame({
        'Title': data.iloc[[i for i, _ in sim_scores]]['Title'].values,
        'Type': data.iloc[[i for i, _ in sim_scores]]['Type'].values,
        'BodyPart': data.iloc[[i for i, _ in sim_scores]]['BodyPart'].values,
        'Equipment': data.iloc[[i for i, _ in sim_scores]]['Equipment'].values,
        'Level': data.iloc[[i for i, _ in sim_scores]]['Level'].values,
        'Desc_vi': data.iloc[[i for i, _ in sim_scores]]['Desc_vi'].values,
        'Similarity': [score for _, score in sim_scores]
    })
    
    # Áp dụng filters
    similar_exercises = apply_all_filters(similar_exercises, injury=injury, equipment=equipment, level=level, type_=type_)
    
    if similar_exercises is None or len(similar_exercises) == 0:
        return None
    
    # Filter theo bodypart nếu có
    if bodypart:
        similar_exercises = similar_exercises[similar_exercises['BodyPart'].str.lower() == bodypart.lower()]
    
    return similar_exercises.head(top_n).reset_index(drop=True)


def recommend_by_bodypart(bodypart, top_n=DEFAULT_TOP_N, type_=None, equipment=None, level=None, injury=None):
    """
    Gợi ý bài tập theo nhóm cơ
    
    Args:
        bodypart (str): Nhóm cơ (abdominals, chest, back, etc.)
        top_n (int): Số lượng bài tập gợi ý
        type_ (str, optional): Loại bài tập
        equipment (str, optional): Thiết bị
        level (str, optional): Cấp độ
        injury (str, optional): Loại chấn thương
        
    Returns:
        pd.DataFrame: Danh sách bài tập được gợi ý
    """
    model = get_model()
    data = model.get_data()
    
    # Lọc theo bodypart
    body_part_data = data[data['BodyPart'].str.lower() == bodypart.lower()]
    
    if body_part_data.empty:
        print(f" Không tìm thấy bài tập thuộc nhóm cơ '{bodypart}'.")
        return None
    
    # Áp dụng filters
    body_part_data = apply_all_filters(body_part_data, injury=injury, equipment=equipment, level=level, type_=type_)
    
    if body_part_data is None or body_part_data.empty:
        return None
    
    # Tính mean similarity trong nhóm
    sub_vectors = np.vstack(body_part_data['bert_vector'].to_numpy())
    sim_matrix = cosine_similarity(sub_vectors)
    mean_scores = sim_matrix.mean(axis=1)
    
    body_part_data = body_part_data.copy()
    body_part_data['Mean_Similarity'] = mean_scores
    
    result = body_part_data.sort_values(by='Mean_Similarity', ascending=False).head(top_n).reset_index(drop=True)
    
    return result[['Title', 'Type', 'BodyPart', 'Equipment', 'Level', 'Desc_vi', 'Mean_Similarity']]


def recommend_by_goal(goal, top_n=DEFAULT_TOP_N, equipment=None, level=None, injury=None):
    """
    Gợi ý bài tập theo mục tiêu
    
    Args:
        goal (str): Mục tiêu (gain abs, big arm, fat lose, etc.)
        top_n (int): Số lượng bài tập gợi ý
        equipment (str, optional): Thiết bị
        level (str, optional): Cấp độ
        injury (str, optional): Loại chấn thương
        
    Returns:
        pd.DataFrame: Danh sách bài tập được gợi ý
    """
    model = get_model()
    data = model.get_data()
    
    goal = goal.lower()
    
    if goal not in GOAL_MAP:
        print(f" Mục tiêu '{goal}' không tồn tại.")
        print(f"Các mục tiêu khả dụng: {', '.join(GOAL_MAP.keys())}")
        return None
    
    bodyparts = GOAL_MAP[goal]["bodyparts"]
    types = GOAL_MAP[goal]["types"]
    
    result = []
    
    # Nếu không có bodyparts cụ thể (ví dụ: fat lose, recover)
    if len(bodyparts) == 0:
        subset = data[data["Type"].isin(types)]
        
        # Áp dụng filters
        subset = apply_all_filters(subset, injury=injury, equipment=equipment, level=level)
        
        if subset is not None and len(subset) > 0:
            if len(subset) > top_n:
                subset = subset.sample(top_n)
            result.append(subset)
    else:
        # Lấy bài tập từ mỗi bodypart
        for bp in bodyparts:
            subset = data[data["BodyPart"] == bp]
            subset = subset[subset["Type"].isin(types)]
            
            # Áp dụng filters
            subset = apply_all_filters(subset, injury=injury, equipment=equipment, level=level)
            
            if subset is None or len(subset) == 0:
                continue
            
            # Lấy ngẫu nhiên 2 bài tập từ mỗi bodypart
            sample_size = min(DEFAULT_SAMPLE_SIZE, len(subset))
            subset = subset.sample(sample_size)
            result.append(subset)
    
    if len(result) == 0:
        print("Không tìm thấy bài nào cho mục tiêu.")
        return None
    
    final = pd.concat(result).reset_index(drop=True)
    
    # Giới hạn số lượng
    if len(final) > top_n:
        final = final.head(top_n)
    
    return final[['Title', 'Type', 'BodyPart', 'Equipment', 'Level', 'Desc_vi']]


def recommend_with_rules(
    mode=None,
    title=None,
    bodypart=None,
    goal=None,
    query=None,
    top_n=DEFAULT_TOP_N,
    injury=None,
    equipment=None,
    level=None,
    type_=None
):
    """
    Hàm gợi ý tổng hợp với nhiều chế độ
    
    Args:
        mode (str): Chế độ gợi ý ('title', 'bodypart', 'goal', 'description')
        title (str, optional): Tên bài tập
        bodypart (str, optional): Nhóm cơ
        goal (str, optional): Mục tiêu
        query (str, optional): Mô tả tự nhiên
        top_n (int): Số lượng bài tập gợi ý
        injury (str, optional): Loại chấn thương
        equipment (str, optional): Thiết bị
        level (str, optional): Cấp độ
        type_ (str, optional): Loại bài tập
        
    Returns:
        pd.DataFrame: Danh sách bài tập được gợi ý
    """
    if mode == "title":
        return recommend_by_title(
            title=title,
            top_n=top_n,
            type_=type_,
            bodypart=bodypart,
            equipment=equipment,
            level=level,
            injury=injury
        )
    
    elif mode == "bodypart":
        return recommend_by_bodypart(
            bodypart=bodypart,
            top_n=top_n,
            type_=type_,
            equipment=equipment,
            level=level,
            injury=injury
        )
    
    elif mode == "goal":
        return recommend_by_goal(
            goal=goal,
            equipment=equipment,
            level=level,
            top_n=top_n,
            injury=injury
        )
    
    elif mode == "description":
        return recommend_from_description(
            user_input=query,
            top_n=top_n,
            injury=injury,
            equipment=equipment,
            level=level
        )
    
    else:
        print(" Sai tên mode. Chọn: 'title', 'bodypart', 'goal', hoặc 'description'")
        return None