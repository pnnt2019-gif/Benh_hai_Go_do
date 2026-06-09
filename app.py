import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Chẩn Đoán Bệnh Gõ Đỏ",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Rút Gọn Để Vừa 1 Trang ---
st.markdown("""
<style>
    /* Thu gọn khoảng cách các khối */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
    h1 { font-size: 1.8rem !important; margin-bottom: 0 !important; padding-bottom: 0 !important; color: #065f46; }
    h3 { font-size: 1.2rem !important; margin-top: 0 !important; color: #0f766e; }
    p { margin-bottom: 0.5rem !important; font-size: 0.95rem !important; }
    
    /* Thiết kế form card hiển thị thông tin gọn gàng */
    .info-card { background-color: #f8fafc; border-left: 4px solid #10b981; padding: 12px 15px; border-radius: 6px; margin-bottom: 10px; font-size: 0.9rem; line-height: 1.4; }
    .warning-card { border-left-color: #f59e0b; background-color: #fffbeb; }
    .danger-card { border-left-color: #ef4444; background-color: #fef2f2; }
    
    /* Khống chế chiều cao hình ảnh để không bị chiếm chỗ */
    img { max-height: 380px !important; object-fit: contain !important; border-radius: 8px; }
    
    /* Căn chỉnh text cột số liệu */
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
# 3. DỮ LIỆU BỆNH
# ==========================================
DISEASE_INFO = {
    "Dom_den": {
        "name": "Bệnh đốm đen",
        "scientific": "Stemphylium sp.",
        "order": "Pleosporales",
        "family": "Pleosporaceae",
        "symptoms": "Vết bệnh cục bộ trên lá, kích thước đa dạng, hình tròn/bầu dục, mang màu đen đặc trưng.",
        "cause": "Nấm Stemphylium sp. tấn công biểu bì lá.",
        "prevention": "Vệ sinh khu vực ươm; tạo thông thoáng; dùng thuốc gốc Đồng hoặc Mancozeb."
    },
    "Chay_la": {
        "name": "Cháy lá sinh lý",
        "scientific": "Abiotic stress",
        "order": "Không",
        "family": "Không",
        "symptoms": "Mô lá khô lại, teo tóp, giòn, màu nâu/xám. Bề mặt nhẵn, KHÔNG có bào tử nấm.",
        "cause": "Yếu tố phi sinh học: sốc nhiệt, gió, muối, ô nhiễm hoặc mất cân bằng dinh dưỡng.",
        "prevention": "Điều chỉnh nước tưới; che lưới cắt nắng; tránh bón thừa đạm (phân hóa học)."
    },
    "Khoe": {
        "name": "Lá Khỏe Mạnh",
        "message": "Cây phát triển tốt. Không phát hiện nấm bệnh hay tổn thương sinh lý trên bề mặt lá Gõ đỏ (Nhóm gỗ I)."
    }
}

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.markdown("<h1>🌿 Ứng Dụng Chẩn Đoán Bệnh Cây Gõ Đỏ</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📸 Hình ảnh đầu vào")
    uploaded_file = st.file_uploader("Chọn ảnh lá cây", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.caption("Khuyến nghị: Chụp rõ bề mặt lá, đủ sáng.")
    st.divider()
    st.markdown("**Thông tin mô hình:**\n- `YOLOv8-seg (1)`: Phân loại\n- `YOLOv8-seg (2)`: Tính tỷ lệ")

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    tab1, tab2 = st.tabs(["🔍 Chẩn Đoán Bệnh", "📊 Tính Cấp Bệnh (Tỷ Lệ)"])
    
    # ---------------------------------------------------------
    # TAB 1: BỐ CỤC 3 CỘT (Ảnh gốc | Ảnh AI | Kết quả rút gọn)
    # ---------------------------------------------------------
    with tab1:
        if st.button("🚀 Chạy Phân Loại (Classification)", type="primary", use_container_width=True):
            if model_chuandoan is not None:
                with st.spinner("Đang phân loại..."):
                    results = model_chuandoan.predict(image_cv, conf=0.8)
                    res = results[0]
                    
                    if len(res.boxes) > 0:
                        c1, c2, c3 = st.columns([1, 1, 1.5])
                        
                        # Cột 1 & 2: Hiển thị hình ảnh
                        with c1:
                            st.image(image_pil, caption="Ảnh gốc đầu vào", use_column_width=True)
                        with c2:
                            res_plotted = res.plot()
                            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="AI Nhận diện", use_column_width=True)
                            
                        # Cột 3: Thông tin nội dung (vừa vặn trên 1 trang)
                        with c3:
                            class_id = int(res.boxes.cls[0].item())
                            conf = float(res.boxes.conf[0].item()) * 100
                            pred_name = res.names[class_id].lower()
                            
                            info_key = "Khoe"
                            if "dom" in pred_name: info_key = "Dom_den"
                            elif "chay" in pred_name: info_key = "Chay_la"
                            
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
                                <div class="info-card"><b>🛡️ Phòng trừ:</b> {info['prevention']}</div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("Mô hình không nhận diện được dấu hiệu với độ tin cậy > 80%.")

    # ---------------------------------------------------------
    # TAB 2: BỐ CỤC 3 CỘT VÀ THUẬT TOÁN AUTO-SORT FOOLPROOF
    # ---------------------------------------------------------
    with tab2:
        if st.button("🚀 Tính Tỷ Lệ & Cấp Bệnh (Segmentation)", type="primary", use_container_width=True):
            if model_capbenh is not None:
                with st.spinner("AI đang nội suy mask..."):
                    results = model_capbenh.predict(image_cv, conf=0.8)
                    res = results[0]
                    
                    if len(res.boxes) > 0 and res.masks is not None:
                        c1, c2, c3 = st.columns([1, 1, 1.2])
                        
                        # Cột 1 & 2: Hình ảnh
                        with c1:
                            st.image(image_pil, caption="Ảnh gốc", use_column_width=True)
                        with c2:
                            res_plotted = res.plot()
                            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Segmentation", use_column_width=True)
                            
                        # Cột 3: Tính toán và Hiển thị
                        with c3:
                            masks = res.masks.data.cpu().numpy()  
                            classes = res.boxes.cls.cpu().numpy() 
                            
                            area_class_0 = 0
                            area_class_1 = 0
                            
                            for i, cls_id in enumerate(classes):
                                if int(cls_id) == 0:
                                    area_class_0 += np.sum(masks[i] > 0.5)
                                elif int(cls_id) == 1:
                                    area_class_1 += np.sum(masks[i] > 0.5)
                            
                            # --- THUẬT TOÁN TỰ ĐỘNG CÂN BẰNG (Chống lỗi ngược ID) ---
                            # Vì vết bệnh luôn nằm trong/nhỏ hơn diện tích toàn bộ lá:
                            # Giá trị lớn hơn CHẮC CHẮN là Toàn Bộ Lá. Giá trị nhỏ hơn là Vết Bệnh.
                            if area_class_0 > 0 and area_class_1 > 0:
                                leaf_pixels = max(area_class_0, area_class_1)
                                disease_pixels = min(area_class_0, area_class_1)
                            elif area_class_0 > 0 or area_class_1 > 0:
                                leaf_pixels = max(area_class_0, area_class_1)
                                disease_pixels = 0  # Chỉ phát hiện được lá (Khỏe mạnh)
                            else:
                                leaf_pixels, disease_pixels = 0, 0
                                
                            st.markdown("### Kết Quả Đo Lường")
                            st.markdown("<div style='margin-bottom: 10px;'>", unsafe_allow_html=True)
                            st.caption(f"📏 Tổng Pixel Lá: {leaf_pixels:,}")
                            st.caption(f"📏 Tổng Pixel Bệnh: {disease_pixels:,}")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            if leaf_pixels > 0:
                                infected_percentage = (disease_pixels / leaf_pixels) * 100
                                infected_percentage = round(infected_percentage, 2)
                            else:
                                infected_percentage = 0.0
                                
                            st.markdown(f"""
                            <div>
                                <div class='metric-label'>Tỷ lệ diện tích tổn thương</div>
                                <div class='metric-value'>{infected_percentage}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Phân loại cấp bệnh
                            level = 0
                            muc_do = "Khỏe mạnh"
                            
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
    st.markdown("""
    <div style="text-align: center; padding: 50px; background-color: #f8fafc; border-radius: 10px; border: 2px dashed #cbd5e1;">
        <h3 style="color: #64748b;">Vui lòng chọn ảnh đầu vào tại thanh menu bên trái</h3>
    </div>
    """, unsafe_allow_html=True)
