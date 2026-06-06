import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Hệ thống Chẩn Đoán Bệnh Cây Trồng",
    page_icon="🌿",
    layout="wide"
)

# ==========================================
# 2. TẢI MÔ HÌNH (Sử dụng Cache để không load lại nhiều lần)
# ==========================================
@st.cache_resource
def load_models():
    # Vui lòng đặt 2 file .pt cùng thư mục với file app.py này
    try:
        model_chuandoan = YOLO('model_chuandoan.pt')
        model_capbenh = YOLO('model_capbenh.pt')
        return model_chuandoan, model_capbenh
    except Exception as e:
        st.error(f"Không tìm thấy mô hình. Lỗi: {e}")
        return None, None

model_chuandoan, model_capbenh = load_models()

# ==========================================
# 3. DỮ LIỆU THÔNG TIN BỆNH
# ==========================================
DISEASE_INFO = {
    "Dom_den": {
        "name": "Bệnh đốm đen",
        "scientific": "Stemphylium sp.",
        "order": "Pleosporales",
        "family": "Pleosporaceae",
        "symptoms": "Xuất hiện các vết bệnh cục bộ trên lá, có kích thước lớn nhỏ khác nhau, hình tròn, bầu dục hoặc bất định hình và mang màu đen đặc trưng.",
        "cause": "Do nấm *Stemphylium* sp. gây ra.",
        "prevention": "- Vệ sinh đồng ruộng, thu gom tiêu hủy lá bệnh.\n- Trồng cây mật độ hợp lý.\n- Dùng thuốc gốc Đồng, Mancozeb..."
    },
    "Chay_la": {
        "name": "Cháy lá sinh lý",
        "scientific": "Abiotic stress",
        "order": "Không có",
        "family": "Không có",
        "symptoms": "Phần mô lá bị ảnh hưởng khô lại, teo tóp, giòn và chuyển sang màu nâu, xám. Bề mặt vết bệnh nhẵn, không có bào tử nấm.",
        "cause": "Các yếu tố bất lợi phi sinh học (môi trường): độ ẩm, gió, muối, chất ô nhiễm, mất cân đối dinh dưỡng.",
        "prevention": "- Điều chỉnh lượng nước tưới.\n- Che lưới cắt nắng.\n- Tránh bón thừa phân hóa học."
    },
    "Khoe": {
        "name": "Lá Khỏe Mạnh",
        "message": "Cây phát triển tốt, không phát hiện dấu hiệu của nấm bệnh hay tổn thương sinh lý."
    }
}

# ==========================================
# 4. THIẾT KẾ GIAO DIỆN CHÍNH
# ==========================================
st.title("🌿 AgriDetect AI - Chẩn Đoán Bệnh Cây Trồng")
st.markdown("Hệ thống nhận diện bệnh hại và tính toán mức độ tổn thương trên lá sử dụng YOLOv8 Segmentation.")

# Tạo 2 tab cho 2 chức năng
tab1, tab2 = st.tabs(["🔍 1. Chẩn đoán bệnh (Phân loại)", "📊 2. Tính cấp bệnh (Segmentation)"])

# Khu vực upload ảnh chung ở thanh bên (Sidebar)
with st.sidebar:
    st.header("Tải ảnh lên")
    uploaded_file = st.file_uploader("Chọn ảnh lá cây (JPG, PNG)", type=["jpg", "jpeg", "png"])
    st.info("Mô hình đang sử dụng:\n- model_chuandoan.pt\n- model_capbenh.pt")

if uploaded_file is not None:
    # Chuyển đổi ảnh upload sang định dạng OpenCV
    image_pil = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # ------------------------------------------
    # TAB 1: CHỨC NĂNG CHẨN ĐOÁN
    # ------------------------------------------
    with tab1:
        st.subheader("Kết quả chẩn đoán bệnh")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image_pil, caption="Ảnh gốc", use_column_width=True)
            
        with col2:
            if st.button("Phân tích bệnh", type="primary", key="btn_chuandoan"):
                if model_chuandoan is not None:
                    with st.spinner("AI đang phân tích..."):
                        # Chạy dự đoán
                        results = model_chuandoan.predict(image_cv)
                        res = results[0]
                        
                        # Vẽ kết quả (Bounding box + Mask) lên ảnh
                        res_plotted = res.plot()
                        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        st.image(res_rgb, caption="Ảnh AI nhận diện", use_column_width=True)
                        
                        # Lấy class_id có độ tin cậy cao nhất (nếu có detect được)
                        if len(res.boxes) > 0:
                            # Lấy ID của hộp đầu tiên được detect
                            class_id = int(res.boxes.cls[0].item())
                            conf = float(res.boxes.conf[0].item()) * 100
                            
                            predicted_name = res.names[class_id] # Lấy tên trực tiếp từ model
                            
                            st.success(f"Phát hiện: **{predicted_name}** (Độ tin cậy: {conf:.2f}%)")
                            
                            # --- HIỂN THỊ THÔNG TIN CHI TIẾT ---
                            # Điều hướng logic để map đúng tên class của bạn vào bộ DATA
                            info_key = "Khoe"
                            if "dom" in predicted_name.lower() or class_id == 0:
                                info_key = "Dom_den"
                            elif "chay" in predicted_name.lower() or class_id == 1:
                                info_key = "Chay_la"
                                
                            info = DISEASE_INFO[info_key]
                            
                            st.markdown(f"### {info['name']}")
                            if info_key != "Khoe":
                                st.markdown(f"**Tên khoa học:** *{info['scientific']}* | **Bộ:** {info['order']} | **Họ:** {info['family']}")
                                st.markdown("#### 🔴 Triệu chứng")
                                st.info(info['symptoms'])
                                st.markdown("#### 🔬 Nguyên nhân")
                                st.warning(info['cause'])
                                st.markdown("#### 🛡️ Biện pháp phòng trừ")
                                st.success(info['prevention'])
                            else:
                                st.info(info['message'])
                                
                        else:
                            st.warning("Mô hình không nhận diện được lá cây hoặc mầm bệnh trong ảnh này.")

    # ------------------------------------------
    # TAB 2: CHỨC NĂNG TÍNH CẤP BỆNH (CHÁY LÁ)
    # ------------------------------------------
    with tab2:
        st.subheader("Phân tích diện tích và tính cấp bệnh")
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.image(image_pil, caption="Ảnh gốc", use_column_width=True)
            
        with col4:
            if st.button("Tính toán cấp bệnh", type="primary", key="btn_capbenh"):
                if model_capbenh is not None:
                    with st.spinner("AI đang nội suy mask và đếm pixel..."):
                        results = model_capbenh.predict(image_cv)
                        res = results[0]
                        
                        res_plotted = res.plot()
                        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        st.image(res_rgb, caption="Ảnh AI phân đoạn (Segmentation)", use_column_width=True)
                        
                        if len(res.boxes) > 0 and res.masks is not None:
                            # Lấy data của masks và class_ids
                            masks = res.masks.data.cpu().numpy()  # Mảng 3D chứa các mask
                            classes = res.boxes.cls.cpu().numpy() # ID của các class được detect
                            
                            # --- QUAN TRỌNG: Đảm bảo khớp ID với file data.yaml khi train ---
                            # Ví dụ: 0 là "Toàn bộ lá" (Leaf), 1 là "Vết bệnh" (Disease)
                            LEAF_CLASS_ID = 0
                            DISEASE_CLASS_ID = 1
                            
                            leaf_pixels = 0
                            disease_pixels = 0
                            
                            # Tính tổng pixel cho từng class dựa trên mask
                            for i, cls_id in enumerate(classes):
                                if int(cls_id) == LEAF_CLASS_ID:
                                    leaf_pixels += np.sum(masks[i])
                                elif int(cls_id) == DISEASE_CLASS_ID:
                                    disease_pixels += np.sum(masks[i])
                            
                            # Tính toán phần trăm (Vết bệnh / Toàn bộ lá)
                            if leaf_pixels > 0:
                                infected_percentage = (disease_pixels / leaf_pixels) * 100
                                infected_percentage = round(infected_percentage, 2)
                            else:
                                infected_percentage = 0.0
                                
                            st.markdown("### Kết quả phân tích Segmentation")
                            st.metric(label="Phần trăm diện tích lá bị tổn thương", value=f"{infected_percentage}%")
                            
                            # Phân loại cấp bệnh theo bảng quy định (Cấp 1-4)
                            level = 0
                            muc_do = "Khỏe mạnh"
                            
                            if infected_percentage > 0:
                                if infected_percentage < 25:
                                    level = 1
                                    muc_do = "Hại nhẹ"
                                elif 25 <= infected_percentage < 50:
                                    level = 2
                                    muc_do = "Hại vừa"
                                elif 50 <= infected_percentage < 75:
                                    level = 3
                                    muc_do = "Hại nặng"
                                else:  # >= 75%
                                    level = 4
                                    muc_do = "Hại rất nặng"
                            
                            if level > 0:
                                st.error(f"⚠️ **Kết luận: BỆNH CẤP {level} ({muc_do})**")
                                # Hiển thị thanh tiến trình
                                st.progress(int(min(infected_percentage, 100)))
                            else:
                                st.success("✅ **Kết luận: Không phát hiện vết bệnh (Cấp 0)**")
                        else:
                            st.success("Không phát hiện vùng tổn thương. Lá khỏe mạnh.")
else:
    st.info("👈 Vui lòng tải ảnh lên từ thanh bên trái để bắt đầu.")