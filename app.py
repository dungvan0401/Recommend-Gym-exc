# app.py
"""
Streamlit App - Hệ thống gợi ý bài tập Gym
"""

import streamlit as st
import pandas as pd
from src.recommend import (
    recommend_from_description,
    recommend_by_title,
    recommend_by_bodypart,
    recommend_by_goal
)
from src.rules import GOAL_MAP, INJURY_MAP
from src.models import get_model

#Cấu hình trang
st.set_page_config(
    page_title="Gym Recommendation",
    layout="wide",
    initial_sidebar_state="expanded"
)

#CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model 
@st.cache_resource
def load_model():
    return get_model()

# Header
st.markdown('<p class="main-header">CÔNG CỤ GỢI Ý BÀI TẬP GYMM</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Tìm kiếm bài tập phù hợp với mục tiêu của bạn</p>', unsafe_allow_html=True)

# Load model
try:
    model = load_model()
    data = model.get_data()
except Exception as e:
    st.error(f"Lỗi khi load dữ liệu: {str(e)}")
    st.stop()

# Sidebar - Filters
st.sidebar.header("BỘ LỌC TÙY CHỌN")

with st.sidebar:
    st.markdown("---")
    
    # Injury filter
    injury_options = ["Không có"] + list(INJURY_MAP.keys())
    injury = st.selectbox(
        "Chấn thương cần tránh:",
        options=injury_options,
        help="Hệ thống sẽ loại bỏ các bài tập không phù hợp"
    )
    injury = None if injury == "Không có" else injury
    
    # Equipment filter
    equipment_list = sorted(data['Equipment'].unique().tolist())
    equipment_options = ["Tất cả"] + equipment_list
    equipment = st.selectbox(
        "Thiết bị:",
        options=equipment_options
    )
    equipment = None if equipment == "Tất cả" else equipment
    
    # Level filter
    level_options = ["Tất cả", "Beginner", "Intermediate", "Advanced"]
    level = st.selectbox(
        "Cấp độ:",
        options=level_options
    )
    level = None if level == "Tất cả" else level
    
    # Top N
    top_n = st.slider(
        "Số lượng bài tập:",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
    
    st.markdown("---")

# Main content - Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    " Tìm theo Mô tả",
    " Tìm theo Tên bài tập",
    " Tìm theo Nhóm cơ",
    " Tìm theo Mục tiêu"
])

# ==================== TAB 1: Tìm theo Mô tả ====================
with tab1:
    st.header("Tìm kiếm bài tập theo mô tả")
    st.markdown("Mô tả bài tập bạn muốn tìm bằng ngôn ngữ tự nhiên")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "Nhập mô tả bài tập:",
            placeholder="Ví dụ: I want exercises for building chest muscles with dumbbells",
            height=100
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn1 = st.button("Tìm kiếm", key="search_desc", use_container_width=True)
    
    if search_btn1 and user_input:
        with st.spinner("Đang tìm kiếm bài tập phù hợp..."):
            results = recommend_from_description(
                user_input=user_input,
                top_n=top_n,
                injury=injury,
                equipment=equipment,
                level=level
            )
            
            if results is not None and len(results) > 0:
                st.success(f"Tìm thấy {len(results)} bài tập phù hợp!")
                
                # Hiển thị kết quả
                for idx, row in results.iterrows():
                    with st.expander(f"**{idx+1}. {row['Title']}** - Độ phù hợp: {row['Similarity']:.2%}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Loại:** {row['Type']}")
                            st.markdown(f"**Nhóm cơ:** {row['BodyPart']}")
                        with col2:
                            st.markdown(f"**Thiết bị:** {row['Equipment']}")
                            st.markdown(f"**Cấp độ:** {row['Level']}")
                        with col3:
                            st.metric("Similarity", f"{row['Similarity']:.2%}")
                        
                        st.markdown("**Mô tả:**")
                        st.info(row['Desc_vi'])
            else:
                st.warning("Không tìm thấy bài tập phù hợp. Thử điều chỉnh bộ lọc!")
    
    elif search_btn1:
        st.warning("Vui lòng nhập mô tả bài tập!")

# ==================== TAB 2: Tìm theo Tên bài tập ====================
with tab2:
    st.header("Tìm bài tập tương tự")
    st.markdown("Chọn một bài tập để tìm các bài tập tương tự")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        title_list = sorted(data['Title'].unique().tolist())
        selected_title = st.selectbox(
            "Chọn bài tập:",
            options=title_list,
            help="Tìm kiếm bài tập tương tự với bài này"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn2 = st.button("Tìm tương tự", key="search_title", use_container_width=True)
    
    if search_btn2 and selected_title:
        with st.spinner("Đang tìm bài tập tương tự..."):
            results = recommend_by_title(
                title=selected_title,
                top_n=top_n,
                equipment=equipment,
                level=level,
                injury=injury
            )
            
            if results is not None and len(results) > 0:
                st.success(f"Tìm thấy {len(results)} bài tập tương tự!")
                
                # Hiển thị kết quả
                for idx, row in results.iterrows():
                    with st.expander(f"**{idx+1}. {row['Title']}** - Độ tương đồng: {row['Similarity']:.2%}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Loại:** {row['Type']}")
                            st.markdown(f"**Nhóm cơ:** {row['BodyPart']}")
                        with col2:
                            st.markdown(f"**Thiết bị:** {row['Equipment']}")
                            st.markdown(f"**Cấp độ:** {row['Level']}")
                        with col3:
                            st.metric("Similarity", f"{row['Similarity']:.2%}")
                        
                        st.markdown("**Mô tả:**")
                        st.info(row['Desc_vi'])
            else:
                st.warning("Không tìm thấy bài tập tương tự. Thử điều chỉnh bộ lọc!")

# ==================== TAB 3: Tìm theo Nhóm cơ ====================
with tab3:
    st.header("Tìm bài tập theo nhóm cơ")
    st.markdown("Chọn nhóm cơ bạn muốn tập luyện")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        bodypart_list = sorted(data['BodyPart'].unique().tolist())
        selected_bodypart = st.selectbox(
            "Chọn nhóm cơ:",
            options=bodypart_list,
            help="Tìm bài tập cho nhóm cơ này"
        )
        
        # Type filter (optional cho tab này)
        type_list = sorted(data['Type'].unique().tolist())
        type_options = ["Tất cả"] + type_list
        type_filter = st.selectbox(
            "Loại bài tập:",
            options=type_options
        )
        type_filter = None if type_filter == "Tất cả" else type_filter
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn3 = st.button("Tìm kiếm", key="search_bodypart", use_container_width=True)
    
    if search_btn3 and selected_bodypart:
        with st.spinner("Đang tìm bài tập..."):
            results = recommend_by_bodypart(
                bodypart=selected_bodypart,
                top_n=top_n,
                type_=type_filter,
                equipment=equipment,
                level=level,
                injury=injury
            )
            
            if results is not None and len(results) > 0:
                st.success(f"Tìm thấy {len(results)} bài tập cho {selected_bodypart}!")
                
                # Hiển thị kết quả
                for idx, row in results.iterrows():
                    with st.expander(f"**{idx+1}. {row['Title']}** - Điểm: {row['Mean_Similarity']:.3f}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Loại:** {row['Type']}")
                            st.markdown(f"**Nhóm cơ:** {row['BodyPart']}")
                        with col2:
                            st.markdown(f"**Thiết bị:** {row['Equipment']}")
                            st.markdown(f"**Cấp độ:** {row['Level']}")
                        with col3:
                            st.metric("Score", f"{row['Mean_Similarity']:.3f}")
                        
                        st.markdown("**Mô tả:**")
                        st.info(row['Desc_vi'])
            else:
                st.warning("Không tìm thấy bài tập phù hợp. Thử điều chỉnh bộ lọc!")

# ==================== TAB 4: Tìm theo Mục tiêu ====================
with tab4:
    st.header("Tìm bài tập theo mục tiêu")
    st.markdown("Chọn mục tiêu tập luyện của bạn")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        goal_options = list(GOAL_MAP.keys())
        goal_descriptions = {
            "gain abs": "Cơ bụng rõ nét",
            "6 packes": "6 múi ",
            "thicked chest": "Ngực dày",
            "broad shoulders": "Vai rộng",
            "big arm": "Tay to",
            "big leg": "Chân to",
            "demon back": "Lưng mặt quỷ",
            "fat lose": "Giảm mỡ",
            "mana": "Tăng sức bền",
            "recover": "Phục hồi"
        }
        
        selected_goal = st.selectbox(
            "Chọn mục tiêu:",
            options=goal_options,
            format_func=lambda x: goal_descriptions.get(x, x)
        )
        
        # Hiển thị thông tin mục tiêu
        if selected_goal:
            goal_info = GOAL_MAP[selected_goal]
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"**Nhóm cơ:** {', '.join(goal_info['bodyparts']) if goal_info['bodyparts'] else 'Toàn thân'}")
            with col_b:
                st.info(f"**Loại:** {', '.join(goal_info['types'])}")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn4 = st.button("Tìm kiếm", key="search_goal", use_container_width=True)
    
    if search_btn4 and selected_goal:
        with st.spinner("Đang tạo kế hoạch tập luyện..."):
            results = recommend_by_goal(
                goal=selected_goal,
                top_n=top_n,
                equipment=equipment,
                level=level,
                injury=injury
            )
            
            if results is not None and len(results) > 0:
                st.success(f" Tìm thấy {len(results)} bài tập cho mục tiêu '{goal_descriptions[selected_goal]}'!")
                
                # Hiển thị kết quả
                for idx, row in results.iterrows():
                    with st.expander(f"**{idx+1}. {row['Title']}**"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Loại:** {row['Type']}")
                            st.markdown(f"**Nhóm cơ:** {row['BodyPart']}")
                        with col2:
                            st.markdown(f"**Thiết bị:** {row['Equipment']}")
                            st.markdown(f"**Cấp độ:** {row['Level']}")
                        
                        st.markdown("**Mô tả:**")
                        st.info(row['Desc_vi'])
            else:
                st.warning("Không tìm thấy bài tập phù hợp. Thử điều chỉnh bộ lọc!")
