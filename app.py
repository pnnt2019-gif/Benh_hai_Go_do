import os
import sys
import subprocess

# ==========================================
# 0. CHUẨN BỊ MÔI TRƯỜNG (CHẠY 1 LẦN DUY NHẤT)
# ==========================================
# Gỡ bản lỗi, cài bản Headless siêu nhẹ và tự động Reset RAM
if not os.path.exists("fixed_cv2.txt"):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        # Đánh dấu đã sửa lỗi xong
        with open("fixed_cv2.txt", "w") as f:
            f.write("done")
        # Ép máy chủ khởi động lại tiến trình Python để xóa cache RAM cũ
        os._exit(0)
    except Exception:
        pass

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Chẩn Đoán Bệnh Gõ Đỏ",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Rút Gọn Để Vừa 1 Trang ---
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
    h1 { font-size: 1.8rem !important; margin-bottom: 0 !important; padding-bottom: 0 !important; color: #065f46; }
    h3 { font-size: 1.2rem !important; margin-top: 0 !important; color: #0f766e; }
    p { margin-bottom: 0.5rem !important; font-size: 0.95rem !important; }
    .info-card { background-color: #f8fafc; border-left: 4px solid #10b981; padding: 12px 15px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9rem; line-height: 1.4; }
    .warning-card { border-left-color: #f59e0b; background-color: #fffbeb; }
    .danger-card { border-left-color: #ef4444; background-color: #fef2f2; }
    img { max-height: 380px !important; object-fit: contain !important; border-radius: 8px; }
    .metric-value { font-size: 2.5rem; font-weight: 800; color: #be123c; line-height: 1.1; }
    .metric-label { font-size: 0.9rem; color: #64748b; font-weight: 600; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. TẢI MÔ HÌNH
# ==========================================
@st.cache_resource
def load_models():
    try:
        model_chuandoan = YOLO('model_chuandoan.pt')
        model_capbenh = YOLO('model_capbenh.pt')
        return model_chuandoan, model_capbenh
    except Exception as e:
        st.error(f"Không tìm thấy mô hình. Lỗi: {e}")
        return None, None

model_chuandoan, model_capbenh = load_models()

# ==========================================
# 3. DỮ LIỆU TỪ ĐIỂN BỆNH HẠI
# ==========================================
DISEASE_INFO = {
    "Dom_den": {
        "name": "Bệnh Đốm Đen",
        "scientific": "Stemphylium sp.",
        "order": "Pleosporales",
        "family": "Pleosporaceae",
        "cause": "Do nấm <i>Stemphylium</i> sp. tấn công biểu bì lá.",
        "symptoms": "Vết bệnh cục bộ trên lá, kích thước đa dạng, hình tròn/bầu dục hoặc bất định hình, mang màu đen đặc trưng.",
        "prevention": "- Sử dụng chế phẩm chứa nấm đối kháng Trichoderma harzianum (2,5 - 3 g/lít nước).\n- Phun ướt đều tán lá định kỳ để ức chế sự nảy mầm và phát triển của bào tử nấm.",
        "image": "dom_den.jpg"
    },
    "Chay_la_sinh_ly": {
        "name": "Cháy Lá Sinh Lý",
        "scientific": "Abiotic stress",
        "order": "Không",
        "family": "Không",
        "cause": "Yếu tố phi sinh học: sốc nhiệt, gió, muối, ô nhiễm hoặc mất cân bằng dinh dưỡng.",
        "symptoms": "Mô lá khô lại, teo tóp, giòn, màu nâu/xám. Bề mặt nhẵn, KHÔNG có dấu hiệu bào tử nấm.",
        "prevention": "- KHÔNG sử dụng thuốc BVTV. Trọng tâm là điều chỉnh vi khí hậu vườn ươm.\n- Trang bị lưới che giảm 50 - 70% ánh sáng, điều chỉnh chế độ tưới tránh gây sốc nước.\n- Phun bổ sung phân bón lá hữu cơ (Amino Acid nồng độ loãng) để giải tỏa stress và kích thích cây phục hồi.",
        "image": "chay_la_sinh_ly.jpg"
    },
    "Chay_la": {
        "name": "Bệnh Cháy Lá",
        "scientific": "Xylella fastidiosa",
        "order": "Lysobacterales",
        "family": "Xanthomonadaceae",
        "cause": "Do vi khuẩn <i>Xylella fastidiosa</i> xâm nhập và làm tắc nghẽn mạch dẫn nước của cây.",
        "symptoms": "Hiện tượng cháy mép lá, thường đi kèm với một dải màu vàng hoặc đỏ rực phân tách rõ rệt giữa phần mô lá còn khỏe mạnh và phần mô đã bị hoại tử.",
        "prevention": "- Phun định kỳ thuốc sát khuẩn gốc Đồng (như Copper Oxychloride, 3 - 5 g/lít nước) hoặc dung dịch phức hợp Đồng - Kẽm.\n- Khi bệnh mới chớm, luân phiên sử dụng thuốc sát khuẩn nội hấp Kasugamycin (1,5 - 2 ml/lít nước) để kìm hãm vi khuẩn nhân lên.",
        "image": "chay_la.jpg"
    },
    "Dom_nau": {
        "name": "Bệnh Đốm Nâu",
        "scientific": "Curvularia sp.",
        "order": "Pleosporales",
        "family": "Pleosporaceae",
        "cause": "Do nấm <i>Curvularia</i> sp. gây ra.",
        "symptoms": "Vết tổn thương cục bộ ở hai mặt lá, hình tròn/bầu dục hoặc bất định. Vùng mô bệnh màu nâu sẫm, rìa ngoài bao quanh bởi quầng sáng màu vàng nhạt phân định rõ ràng.",
        "prevention": "- Ưu tiên chế phẩm sinh học chứa Bacillus subtilis / Trichoderma viride (2 - 3 g/lít nước) hoặc thuốc hóa học chứa Propiconazole (1 ml/lít nước).\n- Phun ướt đều hai mặt lá để ngăn chặn sự xâm nhiễm và kìm hãm nảy mầm bào tử.",
        "image": "dom_nau.jpg"
    },
    "Khoe": {
        "name": "Lá Khỏe Mạnh",
        "message": "Cây phát triển tốt. Không phát hiện nấm bệnh hay tổn thương sinh lý trên bề mặt lá Gõ đỏ (Nhóm gỗ I)."
    }
}

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.markdown("<h1>🌿 Hệ Thống Chẩn Đoán & Tra Cứu Bệnh Gõ Đỏ</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📸 Hình ảnh đầu vào")
    uploaded_file = st.file_uploader("Chọn ảnh lá cây", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.caption("Khuyến nghị: Chụp rõ bề mặt lá, đủ sáng.")
    st.divider()
    st.markdown("**Chức năng hệ thống:**\n1. Chẩn đoán\n2. Tính Mức độ bị hại & Cấp bệnh\n3. Tra cứu")

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
else:
    image_pil = None
    image_cv = None

# Tích hợp 3 chức năng thành 3 Tab cố định
tab1, tab2, tab3 = st.tabs(["🔍 Chẩn Đoán Bệnh", "📊 Tính Cấp Bệnh", "📖 Từ Điển Tra Cứu"])

# ---------------------------------------------------------
# TAB 1: CHẨN ĐOÁN BỆNH
# ---------------------------------------------------------
with tab1:
    if image_cv is not None:
        if st.button("🚀 Chẩn đoán", type="primary", use_container_width=True, key="btn_class"):
            if model_chuandoan is not None:
                with st.spinner("Đang chẩn đoán..."):
                    results = model_chuandoan.predict(image_cv, conf=0.8)
                    res = results[0]
                    
                    if len(res.boxes) > 0:
                        c1, c2, c3 = st.columns([1, 1, 1.5])
                        with c1:
                            st.image(image_pil, caption="Ảnh gốc đầu vào", use_column_width=True)
                        with c2:
                            res_plotted = res.plot(boxes=False, labels=False)
                            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="AI Nhận diện", use_column_width=True)
                            
                        with c3:
                            class_id = int(res.boxes.cls[0].item())
                            conf = float(res.boxes.conf[0].item()) * 100
                            pred_name = res.names[class_id].lower()
                            
                            # Ánh xạ theo model
                            info_key = "Khoe"
                            if "dom" in pred_name: info_key = "Dom_den"
                            elif "chay" in pred_name: info_key = "Chay_la_sinh_ly"
                            
                            info = DISEASE_INFO[info_key]
                            
                            st.markdown(f"### Kết quả: {info['name']}")
                            st.caption(f"Độ tin cậy: {conf:.1f}%")
                            
                            if info_key == "Khoe":
                                st.success(info['message'])
                            else:
                                if info['order'] != "Không":
                                    st.markdown(f"**Khoa học:** <i>{info['scientific']}</i> | **Bộ:** {info['order']} | **Họ:** {info['family']}", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"**Loại:** <i>{info['scientific']}</i> (Yếu tố môi trường)", unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div class="info-card danger-card"><b>🔴 Triệu chứng:</b> {info['symptoms']}</div>
                                <div class="info-card warning-card"><b>🔬 Nguyên nhân:</b> {info['cause']}</div>
                                <div class="info-card"><b>🛡️ Phòng trừ:</b><br>{info['prevention'].replace(chr(10), '<br>')}</div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("Mô hình không nhận diện được dấu hiệu với độ tin cậy > 80%.")
    else:
        st.info("👈 Vui lòng tải ảnh lên ở thanh bên trái để thực hiện Chẩn đoán.")

# ---------------------------------------------------------
# TAB 2: TÍNH TỶ LỆ (SEGMENTATION)
# ---------------------------------------------------------
with tab2:
    if image_cv is not None:
        if st.button("🚀 Tính Mức độ bị hại & Cấp bệnh", type="primary", use_container_width=True, key="btn_seg"):
            if model_capbenh is not None:
                with st.spinner("AI đang nội suy mask..."):
                    results = model_capbenh.predict(image_cv, conf=0.8)
                    res = results[0]
                    
                    if len(res.boxes) > 0 and res.masks is not None:
                        c1, c2, c3 = st.columns([1, 1, 1.2])
                        with c1:
                            st.image(image_pil, caption="Ảnh gốc", use_column_width=True)
                        with c2:
                            res_plotted = res.plot(boxes=False, labels=False)
                            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Segmentation", use_column_width=True)
                            
                        with c3:
                            masks = res.masks.data.cpu().numpy()  
                            classes = res.boxes.cls.cpu().numpy() 
                            
                            total_leaf_mask = np.zeros(masks[0].shape, dtype=bool)
                            disease_mask = np.zeros(masks[0].shape, dtype=bool)
                            
                            for i, cls_id in enumerate(classes):
                                mask_binary = masks[i] > 0.5
                                total_leaf_mask = np.logical_or(total_leaf_mask, mask_binary)
                                name_lower = res.names[int(cls_id)].lower()
                                if "vet" in name_lower:
                                    disease_mask = np.logical_or(disease_mask, mask_binary)
                            
                            leaf_pixels = int(np.sum(total_leaf_mask))
                            disease_pixels = int(np.sum(disease_mask))
                                
                            st.markdown("### Kết Quả Đo Lường")
                            st.markdown("<div style='margin-bottom: 10px;'>", unsafe_allow_html=True)
                            st.caption(f"📏 Tổng Pixel Lá thực tế: {leaf_pixels:,}")
                            st.caption(f"📏 Tổng Pixel Vết Bệnh: {disease_pixels:,}")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            if leaf_pixels > 0:
                                infected_percentage = (disease_pixels / leaf_pixels) * 100
                                infected_percentage = round(infected_percentage, 2)
                            else:
                                infected_percentage = 0.0
                                
                            st.markdown(f"""
                            <div>
                                <div class='metric-label'>Lá bị hại</div>
                                <div class='metric-value'>{infected_percentage}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            level, muc_do = 0, "Khỏe mạnh"
                            if infected_percentage > 0:
                                if infected_percentage < 25: level, muc_do = 1, "Hại nhẹ"
                                elif infected_percentage < 50: level, muc_do = 2, "Hại vừa"
                                elif infected_percentage < 75: level, muc_do = 3, "Hại nặng"
                                else: level, muc_do = 4, "Hại rất nặng"
                            
                            st.progress(int(min(infected_percentage, 100)))
                            if level > 0:
                                st.error(f"⚠️ **Kết luận: BỆNH CẤP {level} ({muc_do})**")
                            else:
                                st.success("✅ **Kết luận: Không phát hiện vết bệnh (Cấp 0)**")
                    else:
                        st.warning("Chưa trích xuất được Mask tổn thương.")
    else:
        st.info("👈 Vui lòng tải ảnh lên ở thanh bên trái để Đo lường cấp bệnh.")

# ---------------------------------------------------------
# TAB 3: TỪ ĐIỂN TRA CỨU
# ---------------------------------------------------------
with tab3:
    st.markdown("### 📖 Hệ Thống Cơ Sở Dữ Liệu Bệnh Hại")
    
    # Tạo menu thả xuống để chọn bệnh
    disease_options = {
        "Bệnh Đốm Đen": "Dom_den",
        "Cháy Lá Sinh Lý": "Chay_la_sinh_ly",
        "Bệnh Cháy Lá (Vi khuẩn)": "Chay_la",
        "Bệnh Đốm Nâu": "Dom_nau"
    }
    
    selected_disease_name = st.selectbox("Chọn loại bệnh để tra cứu chi tiết:", list(disease_options.keys()))
    selected_key = disease_options[selected_disease_name]
    dict_info = DISEASE_INFO[selected_key]
    
    col_dict1, col_dict2 = st.columns([1, 1.2])
    
    with col_dict1:
        # Kiểm tra và tải ảnh từ local folder GitHub
        if os.path.exists(dict_info['image']):
            st.image(dict_info['image'], caption=f"Hình ảnh thực tế: {dict_info['name']}", use_column_width=True)
        else:
            st.info(f"⚠️ Chưa tìm thấy file ảnh `{dict_info['image']}` trên hệ thống.")
            
    with col_dict2:
        st.markdown(f"## {dict_info['name']}")
        
        if dict_info['order'] != "Không":
            st.markdown(f"**Tên khoa học:** <i>{dict_info['scientific']}</i>", unsafe_allow_html=True)
            st.markdown(f"**Bộ:** {dict_info['order']} | **Họ:** {dict_info['family']}")
        else:
            st.markdown(f"**Tên khoa học:** <i>{dict_info['scientific']}</i> (Yếu tố phi sinh học)", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown(f"""
        <div class="info-card warning-card"><b>🔬 Nguyên nhân:</b> {dict_info['cause']}</div>
        <div class="info-card danger-card"><b>🔴 Triệu chứng:</b> {dict_info['symptoms']}</div>
        <div class="info-card"><b>🛡️ Biện pháp phòng trừ:</b><br>{dict_info['prevention'].replace(chr(10), '<br>')}</div>
        """, unsafe_allow_html=True)
