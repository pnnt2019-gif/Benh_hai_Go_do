import os
import pathlib
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

# ==========================================
# 0. HACK FIX: LỖI TƯƠNG THÍCH WINDOWS -> LINUX CHO PYTORCH
# ==========================================
if os.name != 'nt':
    pathlib.WindowsPath = pathlib.PosixPath

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Chẩn Đoán Bệnh Gõ Đỏ",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
@st.cache_resource(show_spinner="Đang khởi tạo AI...")
def load_models():
    try:
        m_chuandoan = YOLO('model_chuandoan.pt')
        m_capbenh = YOLO('model_capbenh.pt')
        return m_chuandoan, m_capbenh
    except Exception as e:
        st.error(f"Lỗi tải mô hình: Vui lòng kiểm tra lại file .pt. Chi tiết lỗi: {e}")
        return None, None

model_chuandoan, model_capbenh = load_models()

# ==========================================
# 3. DỮ LIỆU TỪ ĐIỂN BỆNH HẠI LÂM SINH
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
        "symptoms": "Mô lá khô lại, teo tóp, giòn, màu nâu/xám. Bề mặt nhẵn, không có dấu hiệu bào tử nấm.",
        "prevention": "- Không sử dụng thuốc bảo vệ thực vật. Trọng tâm là điều chỉnh vi khí hậu vườn ươm.\n- Trang bị lưới che giảm 50 - 70% ánh sáng, điều chỉnh chế độ tưới tránh gây sốc nước.\n- Phun bổ sung phân bón lá hữu cơ (Amino Acid nồng độ loãng) để giải tỏa stress và kích thích cây phục hồi.",
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
        "scientific": "Cercospora spp.",
        "order": "Mycosphaerellales",
        "family": "Mycosphaerellaceae",
        "cause": "Do nấm <i>Cercospora</i> sp. gây ra.",
        "symptoms": "Vết tổn thương cục bộ ở hai mặt lá, hình tròn/bầu dục hoặc bất định. Vùng mô bệnh màu nâu sẫm, rìa ngoài bao quanh bởi quầng sáng màu vàng nhạt phân định rõ ràng.",
        "prevention": "- Sử dụng thuốc: Ưu tiên dùng chế phẩm sinh học chứa Bacillus subtilis / Trichoderma spp. (2 - 3 g/lít nước). Nếu bệnh nặng có thể dùng thuốc hóa học Propiconazole (1 ml/lít nước).\n - Cách thức: Phun ướt đều cả hai mặt lá để ngăn chặn nấm xâm nhiễm và kìm hãm bào tử nảy mầm.",
        "image": "dom_nau.jpg"
    },
    "Khoe": {
        "name": "Lá Khỏe Mạnh",
        "message": "Cây phát triển tốt. Không phát hiện nấm bệnh hay tổn thương sinh lý trên bề mặt lá Gõ đỏ (loài gỗ Nhóm I mang giá trị kinh tế và bảo tồn cao)."
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
    st.markdown("**Chức năng hệ thống:**\n1. Chẩn đoán bệnh lý\n2. Phân tích diện tích & Cấp bệnh\n3. Tra cứu dữ liệu")

if uploaded_file is not None:
    # Fix: Áp dụng ImageOps để xoay ảnh đúng chiều EXIF từ điện thoại
    image_pil = Image.open(uploaded_file)
    image_pil = ImageOps.exif_transpose(image_pil)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
else:
    image_pil = None
    image_cv = None

tab1, tab2, tab3 = st.tabs(["🔍 Chẩn Đoán Bệnh", "📊 Phân Tích Cấp Bệnh", "📖 Cơ Sở Dữ Liệu"])

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
                            st.image(image_pil, caption="Ảnh gốc đầu vào", use_container_width=True)
                        with c2:
                            # Fix: Cho phép hiển thị khung bounding box để xác định vị trí nấm bệnh
                            res_plotted = res.plot(conf=True, line_width=2)
                            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="AI Nhận diện", use_container_width=True)
                            
                        with c3:
                            # Fix: Trích xuất loại bệnh dựa trên bounding box có độ tin cậy cao nhất thay vì ngẫu nhiên
                            conf_values = res.boxes.conf.cpu().numpy()
                            best_idx = np.argmax(conf_values)
                            
                            class_id = int(res.boxes.cls[best_idx].item())
                            conf = float(conf_values[best_idx]) * 100
                            pred_name = res.names[class_id].lower()
                            
                            info_key = "Khoe"
                            if "dom" in pred_name: info_key = "Dom_den"
                            elif "chay" in pred_name: info_key = "Chay_la_sinh_ly"
                            
                            info = DISEASE_INFO[info_key]
                            
                            st.markdown(f"### Kết quả: {info['name']}")
                            st.caption(f"Độ tin cậy cao nhất: {conf:.1f}%")
                            
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
                        st.warning("Mô hình không nhận diện được dấu hiệu bệnh lý thực vật (Độ tin cậy > 80%). Có thể lá đang ở trạng thái khỏe mạnh.")
    else:
        st.info("👈 Vui lòng tải ảnh lên ở thanh bên trái để thực hiện Chẩn đoán.")

# ---------------------------------------------------------
# TAB 2: TÍNH TỶ LỆ VÀ CẤP BỆNH (SEGMENTATION)
# ---------------------------------------------------------
with tab2:
    if image_cv is not None:
        if st.button("🚀 Phân Tích Mức Độ Bị Hại", type="primary", use_container_width=True, key="btn_seg"):
            if model_capbenh is not None:
                with st.spinner("AI đang phân tích diện tích vùng tổn thương..."):
                    results = model_capbenh.predict(image_cv, conf=0.8)
                    res = results[0]
                    
                    # Fix: Thêm kiểm tra an toàn đảm bảo masks tồn tại (hasattr) để chống crash
                    if len(res.boxes) > 0 and hasattr(res, 'masks') and res.masks is not None:
                        c1, c2, c3 = st.columns([1, 1, 1.2])
                        with c1:
                            st.image(image_pil, caption="Ảnh gốc", use_container_width=True)
                        with c2:
                            res_plotted = res.plot(boxes=False, labels=False)
                            st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Vùng bị hại", use_container_width=True)
                            
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
                            
                            # Định dạng hiển thị dấu thập phân dạng dấu phẩy theo chuẩn Việt Nam
                            display_percentage = f"{infected_percentage:.2f}".replace('.', ',')
                            
                            # Hiển thị khối Metric duy nhất: Mức độ bị hại
                            st.markdown(f"""
                            <div>
                                <div class='metric-label'>Mức độ bị hại</div>
                                <div class='metric-value'>{display_percentage}%</div>
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
                        st.warning("Hệ thống chưa trích xuất được vùng tổn thương (Mask) trên lá.")
    else:
        st.info("👈 Vui lòng tải ảnh lên ở thanh bên trái để phân tích cấp bệnh.")

# ---------------------------------------------------------
# TAB 3: TỪ ĐIỂN TRA CỨU
# ---------------------------------------------------------
with tab3:
    st.markdown("### 📖 Cơ Sở Dữ Liệu Bệnh Hại Gõ Đỏ")
    
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
        if os.path.exists(dict_info['image']):
            st.image(dict_info['image'], caption=f"Hình ảnh thực tế: {dict_info['name']}", use_container_width=True)
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
