# ------------------------------------------
    # TAB 2: CHỨC NĂNG TÍNH CẤP BỆNH (CHÁY LÁ)
    # ------------------------------------------
    with tab2:
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.image(image_pil, caption="Ảnh gốc", use_column_width=True)
            
        with col4:
            if st.button("Tính toán cấp bệnh", type="primary", key="btn_capbenh"):
                if model_capbenh is not None:
                    with st.spinner("AI đang nội suy mask và đếm pixel..."):
                        results = model_capbenh.predict(image_cv, conf=0.8)
                        res = results[0]
                        
                        res_plotted = res.plot()
                        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                        st.image(res_rgb, caption="Ảnh AI phân đoạn (Segmentation)", use_column_width=True)
                        
                        if len(res.boxes) > 0 and res.masks is not None:
                            masks = res.masks.data.cpu().numpy()  
                            classes = res.boxes.cls.cpu().numpy() 
                            
                            # --- TỰ ĐỘNG DÒ ID THEO TÊN CLASS TRÊN ẢNH ---
                            LEAF_CLASS_ID = None
                            DISEASE_CLASS_ID = None
                            
                            for idx, name in res.names.items():
                                name_lower = name.lower()
                                if "vet" in name_lower:  # Lớp chứa chữ "vet" -> VetChayLa
                                    DISEASE_CLASS_ID = idx
                                else:  # Lớp còn lại -> ChayLa (Toàn bộ chiếc lá)
                                    LEAF_CLASS_ID = idx
                            
                            leaf_pixels = 0
                            disease_pixels = 0
                            
                            # Cộng dồn số pixel từ các mask tương ứng
                            for i, cls_id in enumerate(classes):
                                if LEAF_CLASS_ID is not None and int(cls_id) == LEAF_CLASS_ID:
                                    leaf_pixels += np.sum(masks[i] > 0.5)
                                elif DISEASE_CLASS_ID is not None and int(cls_id) == DISEASE_CLASS_ID:
                                    disease_pixels += np.sum(masks[i] > 0.5)
                            
                            # Hiển thị bảng theo dõi số lượng pixel thực tế để tiện đối chiếu
                            st.markdown("##### 📊 Thống kê điểm ảnh (Pixel Count)")
                            st.text(f"  - Diện tích lớp Lá (ChayLa): {leaf_pixels} pixel")
                            st.text(f"  - Diện tích lớp Vết bệnh (VetChayLa): {disease_pixels} pixel")
                            
                            # Tính toán phần trăm diện tích tổn thương
                            if leaf_pixels > 0:
                                infected_percentage = (disease_pixels / leaf_pixels) * 100
                                infected_percentage = round(infected_percentage, 2)
                            else:
                                infected_percentage = 0.0
                                
                            st.markdown("### Kết quả phân tích Segmentation")
                            st.metric(label="Phần trăm diện tích lá bị tổn thương", value=f"{infected_percentage}%")
                            
                            # Phân loại cấp bệnh dựa theo bảng quy định
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
                                st.progress(int(min(infected_percentage, 100)))
                            else:
                                st.success("✅ **Kết luận: Không phát hiện vết bệnh (Cấp 0)**")
                        else:
                            st.success("Không phát hiện vùng tổn thương (Cấp 0).")
