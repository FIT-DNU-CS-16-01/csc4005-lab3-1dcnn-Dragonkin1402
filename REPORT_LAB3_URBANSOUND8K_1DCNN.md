# BÁO CÁO LAB 3: PHÂN LOẠI ÂM THANH MÔI TRƯỜNG URBANSOUND8K BẰNG 1D-CNN
MÃ SV: 1771040006
Họ Và Tên: Nguyễn Mạnh Cường
Lớp: KHMT1701
REPO: https://github.com/FIT-DNU-CS-16-01/csc4005-lab3-1dcnn-Dragonkin1402

## 1. Mục tiêu bài thực hành

Mục tiêu của bài lab là xây dựng mô hình phân loại âm thanh môi trường đô thị dựa trên bộ dữ liệu **UrbanSound8K**. Khác với bài toán phân loại ảnh sử dụng CNN 2D trên ma trận pixel, bài toán âm thanh xử lý tín hiệu thay đổi theo thời gian. Vì vậy, bài lab tập trung vào việc chuẩn hóa audio, trích xuất đặc trưng âm thanh và sử dụng mô hình **1D-CNN** để học các mẫu cục bộ theo trục thời gian.

Trong bài thực hành này, em triển khai hai cấu hình chính:

- **MFCC + 1D-CNN**: cấu hình baseline.
- **Log-mel + 1D-CNN**: cấu hình mở rộng để so sánh với MFCC.

Mô hình cần có pipeline tiền xử lý audio, huấn luyện, đánh giá, lưu learning curves, confusion matrix, metrics và log kết quả lên W&B.

---

## 2. Bộ dữ liệu sử dụng

Bộ dữ liệu được sử dụng là **UrbanSound8K**, gồm các đoạn âm thanh ngắn trong môi trường đô thị. Dữ liệu có 10 lớp âm thanh:

1. air_conditioner
2. car_horn
3. children_playing
4. dog_bark
5. drilling
6. engine_idling
7. gun_shot
8. jackhammer
9. siren
10. street_music

Cấu trúc dữ liệu sau khi giải nén:

```text
UrbanSound8K/
├── audio/
│   ├── fold1/
│   ├── fold2/
│   ├── ...
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

Trong bài thực hành, dữ liệu được chia theo fold:

```text
Train folds: fold1 đến fold8
Validation fold: fold9
Test fold: fold10
```

---

## 3. Tiền xử lý dữ liệu

Trước khi đưa dữ liệu âm thanh vào mô hình, audio được xử lý qua các bước sau:

### 3.1. Đọc metadata

File `UrbanSound8K.csv` được dùng để lấy thông tin tên file âm thanh, fold chứa file và nhãn lớp tương ứng.

### 3.2. Đọc file âm thanh

Mỗi file `.wav` được đọc từ thư mục:

```text
UrbanSound8K/audio/foldX/
```

Trong đó `X` là số fold tương ứng.

### 3.3. Chuẩn hóa sample rate

Tất cả audio được đưa về cùng sample rate:

```text
sample_rate = 16000 Hz
```

Việc đưa audio về cùng sample rate giúp các file âm thanh có cùng chuẩn lấy mẫu, tránh việc mô hình học trên các tín hiệu có tốc độ lấy mẫu khác nhau.

### 3.4. Chuẩn hóa độ dài audio

Mỗi đoạn audio được đưa về cùng độ dài:

```text
duration = 4.0 giây
```

Nếu audio ngắn hơn 4 giây thì được padding thêm số 0. Nếu audio dài hơn 4 giây thì được cắt bớt. Việc này giúp đầu vào của mô hình có cùng kích thước.

### 3.5. Trích xuất đặc trưng

Hai loại đặc trưng được sử dụng:

#### MFCC

MFCC là đặc trưng âm thanh giúp biểu diễn thông tin phổ theo thời gian một cách gọn hơn so với waveform gốc. Cấu hình sử dụng:

```text
n_mfcc = 40
```

Input đưa vào mô hình có dạng:

```text
[batch_size, n_mfcc, time_frames]
```

#### Log-mel

Log-mel spectrogram giữ lại nhiều thông tin phổ hơn MFCC. Cấu hình sử dụng:

```text
n_mels = 64
```

Input đưa vào mô hình có dạng:

```text
[batch_size, n_mels, time_frames]
```

---

## 4. Mô hình sử dụng

Mô hình sử dụng trong bài là **1D-CNN**.

Kiến trúc tổng quát gồm:

```text
Conv1D
BatchNorm1D
ReLU
MaxPool1D
Dropout
AdaptiveAvgPool1D
Linear Classifier
```

Với dữ liệu âm thanh, Conv1D trượt theo chiều thời gian để học các mẫu âm thanh cục bộ. Ví dụ:

- Tiếng chó sủa thường có các đoạn âm thanh ngắn, rõ.
- Tiếng điều hòa thường đều và kéo dài.
- Tiếng còi xe có thể có pattern cao độ nổi bật.
- Tiếng khoan hoặc jackhammer có pattern lặp lại mạnh.

Do đó, 1D-CNN phù hợp để học sự biến thiên của đặc trưng âm thanh theo thời gian.

---

## 5. Cấu hình thực nghiệm

### 5.1. Cấu hình baseline MFCC + 1D-CNN

| Tham số | Giá trị |
|---|---:|
| Feature type | MFCC |
| Sample rate | 16000 |
| Duration | 4.0 giây |
| n_mfcc | 40 |
| Train folds | 1–8 |
| Validation fold | 9 |
| Test fold | 10 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Dropout | 0.35 |
| Early stopping | Có |
| Logging | W&B |

Lệnh chạy:

```bash
python -m src.train \
--config configs/baseline_mfcc_1dcnn.json \
--data_dir UrbanSound8K \
--run_name mssv_mfcc_1dcnn_baseline
```

### 5.2. Cấu hình mở rộng Log-mel + 1D-CNN

| Tham số | Giá trị |
|---|---:|
| Feature type | Log-mel |
| Sample rate | 16000 |
| Duration | 4.0 giây |
| n_mels | 64 |
| Train folds | 1–8 |
| Validation fold | 9 |
| Test fold | 10 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Dropout | 0.35 |
| Early stopping | Có |
| Logging | W&B |

Lệnh chạy:

```bash
python -m src.train \
--config configs/baseline_mfcc_1dcnn.json \
--data_dir UrbanSound8K \
--feature_type logmel \
--run_name mssv_logmel_1dcnn
```

---

## 6. Kết quả thực nghiệm

### 6.1. Kết quả MFCC + 1D-CNN

Kết quả sau khi train baseline MFCC:

| Chỉ số | Giá trị |
|---|---:|
| Best epoch | 7 |
| Best validation accuracy | 0.5983 |
| Test accuracy | 0.4581 |
| Macro F1 | 0.4596 |

Mô hình MFCC + 1D-CNN đã train thành công và lưu kết quả tại:

```text
outputs/mssv_mfcc_1dcnn_baseline
```

Các file kết quả gồm:

```text
best_model.pt
history.csv
curves.png
confusion_matrix.png
metrics.json
used_config.json
```

### 6.2. Kết quả Log-mel + 1D-CNN

Kết quả sau khi train mô hình Log-mel:

| Chỉ số | Giá trị |
|---|---:|
| Best epoch | 7 |
| Best validation accuracy | 0.6134 |
| Test accuracy | 0.4968 |
| Macro F1 | 0.4942 |

Mô hình Log-mel + 1D-CNN đã train thành công và lưu kết quả tại:

```text
outputs/mssv_logmel_1dcnn
```

---

## 7. So sánh kết quả MFCC và Log-mel

| Mô hình | Best Val Accuracy | Test Accuracy | Macro F1 | Nhận xét |
|---|---:|---:|---:|---|
| MFCC + 1D-CNN | 0.5983 | 0.4581 | 0.4596 | Baseline chạy ổn |
| Log-mel + 1D-CNN | 0.6134 | 0.4968 | 0.4942 | Tốt hơn MFCC |

Qua bảng kết quả, mô hình **Log-mel + 1D-CNN** cho kết quả tốt hơn mô hình **MFCC + 1D-CNN** ở cả ba chỉ số chính:

```text
Validation accuracy: 0.5983 → 0.6134
Test accuracy: 0.4581 → 0.4968
Macro F1: 0.4596 → 0.4942
```

Điều này cho thấy đặc trưng Log-mel có thể giữ lại nhiều thông tin phổ âm thanh hơn MFCC, từ đó giúp mô hình học tốt hơn trong bài toán phân loại âm thanh môi trường.

---

## 8. Phân tích learning curves

Dựa trên log huấn luyện, mô hình có khả năng học được từ dữ liệu train. Tuy nhiên, có xuất hiện dấu hiệu **overfitting nhẹ**.

Ở mô hình MFCC:

```text
train_acc ≈ 0.7992
val_acc ≈ 0.5659
test_acc ≈ 0.4581
```

Ở mô hình Log-mel:

```text
train_acc ≈ 0.7825
val_acc ≈ 0.5702
test_acc ≈ 0.4968
```

Train accuracy cao hơn validation accuracy và test accuracy khá rõ. Điều này cho thấy mô hình học tốt trên tập train nhưng khả năng tổng quát sang dữ liệu mới chưa thật sự cao.

Nguyên nhân có thể là:

- Dữ liệu âm thanh đô thị có nhiều nhiễu nền.
- Một số lớp âm thanh có đặc điểm tương tự nhau.
- Số lượng mẫu train trong cấu hình lab bị giới hạn.
- Mô hình có thể ghi nhớ một phần dữ liệu train.
- Một số âm thanh trong test set có môi trường hoặc cường độ khác với train set.

Tuy vậy, đây là kết quả hợp lý trong phạm vi bài lab vì mục tiêu chính là xây dựng pipeline xử lý audio và đánh giá mô hình, không phải tối ưu accuracy cao nhất.

---

## 9. Phân tích confusion matrix

Confusion matrix được lưu tại:

```text
outputs/mssv_mfcc_1dcnn_baseline/confusion_matrix.png
outputs/mssv_logmel_1dcnn/confusion_matrix.png
```

Confusion matrix giúp quan sát mô hình dự đoán đúng và sai ở từng lớp. Với bài toán UrbanSound8K, một số lớp có thể dễ bị nhầm lẫn do đặc điểm âm thanh gần nhau.

Các cặp lớp có khả năng bị nhầm:

| Lớp âm thanh | Có thể bị nhầm với | Nguyên nhân |
|---|---|---|
| drilling | jackhammer | Đều là âm thanh máy móc, có pattern lặp mạnh |
| air_conditioner | engine_idling | Đều có âm thanh nền đều, kéo dài |
| children_playing | street_music | Có thể chứa nhiều tiếng nền phức tạp |
| siren | car_horn | Đều là âm thanh giao thông, cường độ cao |
| dog_bark | children_playing | Có thể xuất hiện trong môi trường nhiều tiếng ồn |

Việc mô hình nhầm giữa các lớp này là hợp lý vì âm thanh trong môi trường thực tế thường có nhiễu, âm lượng khác nhau và có thể chồng lẫn nhiều nguồn âm.

---

## 10. Vai trò của W&B

Trong bài lab, W&B được sử dụng để theo dõi quá trình huấn luyện. Các chỉ số được log gồm:

```text
train_loss
val_loss
train_acc
val_acc
epoch_time_sec
test_accuracy
test_loss
test_macro_f1
best_val_accuracy
```

W&B giúp quan sát learning curves, so sánh các lần chạy khác nhau và kiểm tra mô hình có overfitting hay không. Trong bài này, em đã log được cả hai run:

```text
mssv_mfcc_1dcnn_baseline
mssv_logmel_1dcnn
```

Run Log-mel có kết quả tốt hơn nên được chọn là cấu hình tốt hơn trong hai thử nghiệm.

---

## 11. Vì sao MFCC/Log-mel ổn định hơn Raw Waveform?

Raw waveform là tín hiệu âm thanh gốc. Với audio dài 4 giây và sample rate 16000 Hz, mỗi mẫu có khoảng:

```text
4 × 16000 = 64000 điểm tín hiệu
```

Nếu đưa trực tiếp waveform vào mô hình, input rất dài và mô hình phải tự học nhiều đặc trưng từ đầu như biên độ, tần số, nhịp, nhiễu nền và pattern lặp lại.

Trong khi đó, MFCC và Log-mel đã tóm tắt thông tin phổ âm thanh theo thời gian. Vì vậy:

- Đầu vào gọn hơn.
- Mô hình học nhanh hơn.
- Phù hợp với máy cá nhân.
- Giảm độ phức tạp của bài toán.
- Dễ tạo baseline ổn định hơn.

Vì vậy, trong phạm vi bài lab, việc sử dụng MFCC hoặc Log-mel là hợp lý hơn so với raw waveform.

---

## 12. Chọn mô hình tốt nhất

Dựa trên kết quả validation và test, mô hình tốt hơn là:

```text
Log-mel + 1D-CNN
```

Lý do chọn:

- Best validation accuracy cao hơn MFCC.
- Test accuracy cao hơn MFCC.
- Macro F1 cao hơn MFCC.
- Thời gian train mỗi epoch vẫn hợp lý, khoảng 7–8 giây.
- Kết quả cho thấy Log-mel biểu diễn thông tin âm thanh tốt hơn trong thí nghiệm này.

Kết quả mô hình tốt nhất:

| Chỉ số | Giá trị |
|---|---:|
| Model | Log-mel + 1D-CNN |
| Best epoch | 7 |
| Best validation accuracy | 0.6134 |
| Test accuracy | 0.4968 |
| Macro F1 | 0.4942 |

---

## 13. Kết luận

Trong bài lab này, em đã xây dựng thành công pipeline phân loại âm thanh môi trường UrbanSound8K bằng mô hình 1D-CNN. Dữ liệu âm thanh được chuẩn hóa về cùng sample rate, cùng độ dài, sau đó trích xuất đặc trưng MFCC và Log-mel để đưa vào mô hình.

Kết quả thực nghiệm cho thấy mô hình **MFCC + 1D-CNN** đạt test accuracy **45.81%**, trong khi mô hình **Log-mel + 1D-CNN** đạt test accuracy **49.68%**. Như vậy, Log-mel cho kết quả tốt hơn MFCC trong bài thực hành này.

Tuy mô hình vẫn có dấu hiệu overfitting nhẹ, kết quả đạt được là hợp lý với phạm vi bài lab. Bài thực hành giúp em hiểu rõ hơn quy trình xử lý âm thanh, cách đưa audio về đặc trưng theo thời gian, cách sử dụng 1D-CNN cho audio classification, cũng như cách đọc learning curves và confusion matrix để đánh giá mô hình.

Trong các hướng cải thiện tiếp theo, có thể thử:

- Tăng số lượng dữ liệu train.
- Dùng data augmentation cho audio.
- Điều chỉnh dropout.
- Thử learning rate nhỏ hơn.
- Tăng số epoch nhưng kết hợp early stopping.
- Thử kiến trúc CNN sâu hơn.
- So sánh thêm với raw waveform hoặc các mô hình mạnh hơn.

---

## 14. Tài liệu/kết quả nộp kèm

Các file cần nộp:

```text
README.md
REPORT_TEMPLATE.md hoặc file báo cáo Word/PDF
outputs/mssv_mfcc_1dcnn_baseline/curves.png
outputs/mssv_mfcc_1dcnn_baseline/confusion_matrix.png
outputs/mssv_mfcc_1dcnn_baseline/metrics.json
outputs/mssv_logmel_1dcnn/curves.png
outputs/mssv_logmel_1dcnn/confusion_matrix.png
outputs/mssv_logmel_1dcnn/metrics.json
Ảnh hoặc link W&B dashboard
```

Kết quả W&B đã có hai run:

```text
mssv_mfcc_1dcnn_baseline
mssv_logmel_1dcnn
```
