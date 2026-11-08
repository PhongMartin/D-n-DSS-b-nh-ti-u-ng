# Hệ hỗ trợ quyết định đánh giá nguy cơ tiền tiểu đường và tiểu đường

Đồ án INS305 – DSS, Nhóm 01. Dữ liệu: CDC BRFSS 2015 "Diabetes Health Indicators" (253.680 bản ghi, 21 đặc trưng, nhãn `Diabetes_binary`; 1 = tiền tiểu đường hoặc tiểu đường, **tự khai**).

**Trạng thái:** Giai đoạn 1 và 2 đã chạy xong (trong `DSS_FinalProject.ipynb`, mục 1.1 đến 2.10). Giai đoạn 3 (mục 3.0 đến 3.8) và giao diện `app.py` đã có mã; cần chạy notebook một lần để sinh kết quả và `models/Phase3_system.joblib`.

## Tệp trong gói
| Tệp | Vai trò |
|---|---|
| `DSS_FinalProject.ipynb` | Notebook duy nhất cho Giai đoạn 1, 2 và 3 (không còn `DSS_Phase2.ipynb`) |
| `dss_utils.py` | Mã dùng chung Giai đoạn 1–2: đọc dữ liệu và danh sách tập, mô hình, tinh chỉnh, chỉ số, hiệu chuẩn, ngưỡng |
| `dss_stage3.py` | Mã Giai đoạn 3: chốt hiệu chuẩn, phân mức, Top-k, bỏ sót theo nhóm, SHAP, Decision Engine, What-if, so sánh cấu hình |
| `app.py` | Giao diện Streamlit (8 chức năng: Dashboard, Risk Assessment, Explainable AI, Decision Support, What-if Analysis, Capacity Planning, Model Evaluation, Export Results) |
| `ui.py` | Thành phần giao diện (CSS, đồng hồ đo, thẻ quy tắc); đặt cùng thư mục với `app.py` |
| `export_report.py` | Tạo file xuất: Excel nhiều trang đã định dạng và báo cáo HTML/PDF |
| `rules/decision_rules.json` | Bộ 16 quy tắc (10 guideline + 6 operational), mỗi quy tắc có nguồn, tài liệu/mục, số khuyến cáo và trạng thái xác minh |
| `rules/test_cases.csv` | 23 tình huống kiểm thử cho bộ quy tắc (ngưỡng sát biên lấy theo ngưỡng thật) |
| `tests/test_end_to_end.py` | Kiểm thử xuyên suốt hồ sơ → mô hình → mức → quy tắc → SHAP → file xuất; đọc ngưỡng từ `Phase3_system.joblib` |
| `tools/clean.py` | Xóa `.ipynb_checkpoints`, `__pycache__`, file tạm trước khi nộp |
| `requirements.txt`, `README.md` | Thư viện và hướng dẫn |

## Cách chạy
1. `pip install -r requirements.txt`
2. Đặt `diabetes_binary_health_indicators_BRFSS2015.csv` vào thư mục `datas/` (hoặc cùng thư mục notebook). Có thể đặt biến môi trường `DSS_PROJECT_DIR` để chỉ định thư mục dự án; không cần sửa đường dẫn.
3. Mở `DSS_FinalProject.ipynb`, chạy tuần tự (Run All). Giai đoạn 2 chạy lâu (khoảng 45 phút so sánh chiến lược mất cân bằng và 39 phút đánh giá cuối); Giai đoạn 3 huấn luyện lại LightGBM cho 5 seed (vài phút mỗi seed, được lưu vào `results/phase3/` để lần sau chạy nhanh).
4. Chạy giao diện: `streamlit run app.py` (sau khi đã chạy xong mục 3.5 và 3.6 của notebook).
5. Dashboard, Capacity Planning và Model Evaluation đọc số liệu từ `results/tables/*.csv` do notebook sinh ra (không gõ tay).
6. Kiểm thử: `python tests/test_end_to_end.py` (cần có `models/Phase3_system.joblib`); kết quả lưu ở `results/tables/5_e2e_test_results.csv`.

### Chạy trên máy mới (môi trường Python sạch)
```
python -m venv .venv
.venv\Scripts\activate            # Windows;  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # đã gồm pyarrow (bắt buộc để nạp Phase3_system.joblib)
python tests/test_end_to_end.py
streamlit run app.py
```
Nên chạy Streamlit bằng đúng Python đã cài `requirements.txt` (`python -m streamlit run app.py`). Nếu báo `No module named 'pyarrow'` hoặc `'lightgbm'`, chạy `python -m pip install -r requirements.txt` bằng chính Python đó.

## Cấu hình đã chốt (Giai đoạn 3)
- **Mô hình:** LightGBM; siêu tham số và chiến lược mất cân bằng lấy từ `models/LightGBM_bundle.joblib` (Giai đoạn 2). Mỗi seed 42–46 có một mô hình riêng, huấn luyện trên train của seed đó.
- **Hiệu chuẩn:** chọn bằng quy tắc đặt trước trên **validation** (mục 3.0): chỉ dùng Platt/Isotonic nếu giảm Brier ở mọi seed và mức giảm trung bình ≥ 0,0001; nếu không, giữ xác suất chưa hiệu chuẩn. Mô hình triển khai dùng đúng cấu hình này (in ở cuối mục 3.0 và lưu trong `models/Phase3_system.joblib`, trường `calibration`).
- **Ngưỡng:** học trên validation. Phân mức sơ đồ A: Thấp < ngưỡng Recall ≥ 80% ≤ Trung bình < ngưỡng F1 lớn nhất ≤ Cao.
- **Không dùng test** để chọn mô hình, hiệu chuẩn, ngưỡng hay sơ đồ phân mức; test chỉ để báo cáo.

## Sử dụng các tệp mô hình đã lưu
```python
import dss_stage3 as S3
system = S3.load_system()            # models/Phase3_system.joblib: mô hình, bộ hiệu chuẩn, ngưỡng, 21 biến
rules = S3.load_rules()              # rules/decision_rules.json
profile = S3._neutral(); profile.update(Age=8, BMI=33.0, HighBP=1)   # 21 biến, mã theo BRFSS
result = S3.assess(system, profile, rules)   # xác suất, mức nguy cơ, quy tắc kích hoạt, SHAP
what = S3.what_if(system, profile, {"BMI": 30.5, "PhysActivity": 1}, rules)
```
`models/*_bundle.joblib` (Giai đoạn 2) chứa mô hình, bộ hiệu chuẩn và ngưỡng của từng thuật toán (seed 42).

## Lưu ý diễn giải
- Xác suất là khả năng thuộc nhóm có tiền sử được chẩn đoán theo dữ liệu khảo sát 2015, không phải xác suất mắc bệnh trong tương lai; hệ thống không chẩn đoán và không thay thế nhân viên y tế. Mức nguy cơ thấp không có nghĩa chắc chắn không mắc bệnh.
- SHAP nằm trên thang log-odds của đầu ra **thô** của mô hình (trước hiệu chuẩn).
- What-if chỉ mô phỏng phản ứng của mô hình, không chứng minh hiệu quả của hành động. `Smoker` nghĩa là đã từng hút ≥ 100 điếu, không được đổi 1 → 0.
- Năng lực xử lý (100/200/300/400 hồ sơ trên 1.000 người) và tỉ lệ chi phí FN:FP là kịch bản giả định.
- Nguồn quy tắc (ADA Standards of Care 2026, USPSTF 2021, CDC National DPP) mới đối chiếu qua nguồn thứ cấp. Trường `rec_no` (số khuyến cáo) để trống và `status` ghi rõ “chưa đối chiếu bản gốc” cho tới khi nhóm mở bản gốc và điền.
- R05 dùng mã tuổi 4–10 (35–69 tuổi) vì BRFSS chỉ có nhóm tuổi 5 năm, không tách được người đúng 70 tuổi (xem `boundary_note` trong `decision_rules.json`).

## Đóng gói sản phẩm cuối
Xóa thư mục `.ipynb_checkpoints`, `__pycache__` và các file tạm trước khi nộp.
