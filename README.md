# THÔNG THIÊN PHIÊN GIẢ

Bảo pháp "Thông Thiên Phiên Giả" là một đại pháp khí tối thượng, dựa trên huyền thuật **Transformer**, lĩnh hội từ cổ thư **[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)**. Nếu đạo hữu thấy pháp bảo này hữu dụng, xin hãy lưu lại một dấu ấn (Star) để thể hiện lòng tán thưởng!

## I. Giải Nghĩa Đại Pháp
- **Bí kíp truyền thừa**:
    - Bí mật của **Biến Hóa Ngữ Giới**: [Tại đây](https://drive.google.com/file/d/182rTpgUdTjDgw4LrAM6ah2B_Iw_4rXQW/view?usp=sharing)
    - **Thiên Đạo Phiên Dịch** (Đang cập nhật)

## II. Tác Giả
- **GitHub**: bangoc123
- **Linh Truyền**: protonxai@gmail.com

Bảo pháp này thuộc về đại môn phái **[Papers-Videos-Code](https://docs.google.com/document/d/1bjmwsYFafizRXlZyJFazd5Jcr3tqpWSiHLvfllWRQBc/edit?usp=sharing)**, nơi tập hợp những kỳ thư thuật pháp của **Trí Tuệ Nhân Tạo**, truyền bá qua kênh **[ProtonX Youtube](https://www.youtube.com/c/ProtonX/videos)**.

---

## III. Kiến Trúc Đại Trận

![image](https://storage.googleapis.com/protonx-cloud-storage/transformer/architecture.PNG)

**[Chú ý] Đạo hữu có thể sử dụng dữ liệu riêng để bồi dưỡng đại pháp này.**

---

## IV. Cách Thiết Lập Pháp Trường

1. Đạo hữu cần phải có **Miniconda**. Nếu chưa có, hãy xem bí kíp hướng dẫn [tại đây](https://conda.io/en/latest/user-guide/install/index.html#regular-installation).
2. Đi đến **pháp đàn** `transformer` và thi triển thần chú:
   ```bash
   conda env create -f environment.yml
   ```
3. Khởi động đại pháp:
   ```bash
   conda activate transformer
   ```

---

## V. Chuẩn Bị Nguyên Liệu

Để luyện thành bảo pháp, cần có **ngữ liệu** bao gồm hai cuốn **chân kinh**:
- **train.en** (Anh ngữ)
- **train.vi** (Việt ngữ)

Ví dụ minh họa:
| train.en   | train.vi      |
|------------|--------------|
| I love you | Tôi yêu bạn  |
| ...        | ....         |

Có thể tham khảo **ngữ liệu giả lập** trong thư mục `./data/mock`.

---

## VI. Bắt Đầu Tu Luyện

Thi triển thần chú để khai mở đại trận:
```bash
python train.py --epochs ${epochs} --input-lang en --target-lang vi --input-path ${path_to_en_text_file} --target-path ${path_to_vi_text_file}
```

Ví dụ: Đạo hữu muốn luyện **Anh-Việt đại pháp** trong **10 chu thiên**:
```bash
python train.py --epochs 10 --input-lang en --target-lang vi --input-path ./data/mock/train.en --target-path ./data/mock/train.vi
```

Những tham số quan trọng cần lưu tâm:
- `input-lang`: Ngôn ngữ nhập vào (VD: en)
- `target-lang`: Ngôn ngữ đích (VD: vi)
- `input-path`: Đường dẫn đến bản kinh nhập vào
- `target-path`: Đường dẫn đến bản kinh mục tiêu
- `model-folder`: Nơi lưu giữ kết quả đại pháp
- `batch-size`: Quy mô mỗi lô luyện công
- `max-length`: Giới hạn độ dài của pháp văn
- `num-examples`: Số lượng câu cần luyện
- `d-model`: Kích thước không gian pháp thuật
- `n`: Số tầng trong Kiến Trúc Đại Pháp
- `h`: Số đầu của "Đa Thần Tỉnh" (Multi-Head Attention)
- `dropout-rate`: Tỷ lệ thất thoát linh khí

Sau khi hoàn thành luyện pháp, **trận pháp** sẽ được lưu tại thư mục đã chỉ định.

---


